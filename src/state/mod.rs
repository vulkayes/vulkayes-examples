use std::{num::NonZeroU32, sync::mpsc, time::Instant};

use winit::window::Window;

use vulkayes_core::{
	ash::{version::DeviceV1_0, vk},
	log
};
use vulkayes_window::winit::winit;

use vulkayes_core::{
	memory::device::naive::NaiveDeviceMemoryAllocator,
	prelude::*,
	resource::image::params::ImageViewRange
};

pub mod frame;
pub mod input;
pub mod setup;

use crate::{dirty_mark, state::frame::CommandState};
use input::InputEvent;
use std::ops::Deref;
use vulkayes_core::swapchain::SwapchainCreateInfo;

// Controls whether `NextFrame::submit_present` waits for the queue operations to finish. Useful for benchmarks.
pub const WAIT_AFTER_FRAME: bool = true;

pub trait ChildExampleState {
	fn render_pass(&self) -> vk::RenderPass;
}

#[derive(Debug)]
pub struct BaseState {
	pub instance: Vrc<Instance>,
	pub device: Vrc<Device>,
	pub present_queue: Vrc<Queue>
}
#[derive(Debug)]
pub struct SwapchainState {
	pub create_info: SwapchainCreateInfo<[u32; 1]>,
	pub swapchain: Vrc<Swapchain>,
	pub present_views: Vec<Vrc<ImageView>>,

	render_pass: Option<vk::RenderPass>,
	pub framebuffers: Vec<vk::Framebuffer>,

	pub scissors: vk::Rect2D,
	pub viewport: vk::Viewport,

	outdated: u8,
	was_recreated: bool
}
impl Drop for SwapchainState {
	fn drop(&mut self) {
		unsafe {
			for framebuffer in self.framebuffers.drain(..) {
				self.swapchain
					.device()
					.destroy_framebuffer(framebuffer, None);
			}
		}
	}
}
pub struct ApplicationState {
	pub base: BaseState,
	pub device_memory_allocator: NaiveDeviceMemoryAllocator,

	pub swapchain: SwapchainState,
	pub surface_size: [NonZeroU32; 2],
	pub depth_image_view: Vrc<ImageView>,

	pub command: frame::CommandState,

	input_receiver: mpsc::Receiver<InputEvent>,

	last_before: Option<Instant>,
	last_before_middle: Option<[Instant; 2]>
}
impl ApplicationState {
	fn create_present_views(images: Vec<Vrc<SwapchainImage>>) -> Vec<Vrc<ImageView>> {
		images
			.into_iter()
			.map(|image| {
				ImageView::new(
					image.into(),
					ImageViewRange::Type2D(0, vulkayes_core::NONZEROU32_ONE, 0),
					None,
					Default::default(),
					vk::ImageAspectFlags::COLOR,
					Default::default()
				)
				.expect("Could not create image view")
			})
			.collect()
	}

	fn create_depth_image_view(
		queue: &Queue,
		size: ImageSize,
		device_memory_allocator: &NaiveDeviceMemoryAllocator,
		command: &mut CommandState
	) -> Vrc<ImageView> {
		let depth_image_view = {
			let depth_image = Image::new(
				queue.device().clone(),
				vk::Format::D16_UNORM,
				size.into(),
				Default::default(),
				vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
				queue.into(),
				vulkayes_core::resource::image::params::AllocatorParams::Some {
					allocator: device_memory_allocator,
					requirements: vk::MemoryPropertyFlags::DEVICE_LOCAL
				},
				Default::default()
			)
			.expect("Could not create depth image");

			ImageView::new(
				depth_image.into(),
				ImageViewRange::Type2D(0, vulkayes_core::NONZEROU32_ONE, 0),
				None,
				Default::default(),
				vk::ImageAspectFlags::DEPTH,
				Default::default()
			)
			.expect("Chould not create depth image view")
		};

		command.execute_setup_blocking(queue, |command_buffer| {
			let cb_lock = command_buffer.lock().expect("vutex poisoned");

			let layout_transition_barriers = vk::ImageMemoryBarrier::builder()
				.image(*depth_image_view.image().deref().deref())
				.old_layout(vk::ImageLayout::UNDEFINED)
				.dst_access_mask(
					vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
						| vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
				)
				.new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
				.subresource_range(depth_image_view.subresource_range().into());

			unsafe {
				command_buffer.pool().device().cmd_pipeline_barrier(
					*cb_lock,
					vk::PipelineStageFlags::BOTTOM_OF_PIPE,
					vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
					vk::DependencyFlags::empty(),
					&[],
					&[],
					&[layout_transition_barriers.build()]
				);
			}
		});

		depth_image_view
	}

	fn create_framebuffers<'a>(
		renderpass: vk::RenderPass,
		present_image_views: impl Iterator<Item = &'a Vrc<ImageView>>,
		depth_image_view: &Vrc<ImageView>
	) -> Vec<vk::Framebuffer> {
		let framebuffers: Vec<vk::Framebuffer> = present_image_views
			.map(|present_image_view| {
				let framebuffer_attachments = [
					*present_image_view.deref().deref(),
					*depth_image_view.deref().deref()
				];
				let frame_buffer_create_info = vk::FramebufferCreateInfo::builder()
					.render_pass(renderpass)
					.attachments(&framebuffer_attachments)
					.width(present_image_view.subresource_image_size().width().get())
					.height(present_image_view.subresource_image_size().height().get())
					.layers(1);

				vulkayes_core::log::debug!(
					"Creating framebuffers for {:#?}",
					framebuffer_attachments
				);
				unsafe {
					present_image_view
						.image()
						.device()
						.create_framebuffer(&frame_buffer_create_info, None)
						.unwrap()
				}
			})
			.collect();

		framebuffers
	}

	/// Checks `self.swapchain.outdated` and attempts to recreate the swapchain.
	fn check_and_recreate_swapchain(&mut self) {
		self.swapchain.was_recreated = false;
		if self.swapchain.outdated != 0 {
			self.swapchain.create_info.image_info.image_size = ImageSize::new_2d(
				self.surface_size[0],
				self.surface_size[1],
				vulkayes_core::NONZEROU32_ONE,
				MipmapLevels::One()
			);
			let new_data = self
				.swapchain
				.swapchain
				.recreate(self.swapchain.create_info, Default::default());

			match new_data {
				Err(e) => {
					log::error!("Could not recreate swapchain {}", e);
					if self.swapchain.outdated > 200 {
						panic!("Could not recreate swapchain for 200 frames");
					}
				}
				Ok(data) => {
					self.swapchain.swapchain = data.swapchain;
					self.swapchain.present_views = Self::create_present_views(data.images);
					self.depth_image_view = Self::create_depth_image_view(
						&self.base.present_queue,
						self.swapchain.create_info.image_info.image_size.into(),
						&self.device_memory_allocator,
						&mut self.command
					);

					self.swapchain.scissors = vk::Rect2D {
						offset: vk::Offset2D { x: 0, y: 0 },
						extent: vk::Extent2D {
							width: self.surface_size[0].get(),
							height: self.surface_size[1].get()
						}
					};
					self.swapchain.viewport = vk::Viewport {
						x: 0.0,
						y: 0.0,
						width: self.surface_size[0].get() as f32,
						height: self.surface_size[1].get() as f32,
						min_depth: 0.0,
						max_depth: 1.0
					};
					self.swapchain.framebuffers = Self::create_framebuffers(
						self.swapchain.render_pass.unwrap(),
						self.swapchain.present_views.iter(),
						&self.depth_image_view
					);

					self.swapchain.outdated = 0;
					self.swapchain.was_recreated = true;
				}
			}
		}
	}

	fn after_setup<T: ChildExampleState>(&mut self, child: &mut T) {
		self.swapchain.render_pass = Some(child.render_pass());
		self.swapchain.framebuffers = Self::create_framebuffers(
			self.swapchain.render_pass.unwrap(),
			self.swapchain.present_views.iter(),
			&self.depth_image_view
		);
	}

	pub fn new_input_thread<T: ChildExampleState>(
		window_size: [NonZeroU32; 2],
		setup_fn: impl FnOnce(&mut Self) -> T + Send + 'static,
		mut render_loop_fn: impl FnMut(&mut Self, &mut T) + Send + 'static,
		cleanup_fn: impl FnOnce(&mut Self, T) + Send + 'static
	) {
		Self::new_input_thread_inner(window_size, move |window, input_receiver, oneoff_sender| {
			let before = Instant::now();
			let mut me = Self::new(window, input_receiver, oneoff_sender);
			let mut t = setup_fn(&mut me);
			me.after_setup(&mut t);
			let after = Instant::now().duration_since(before);
			println!("SETUP: {}", after.as_nanos());

			unsafe {
				dirty_mark::init();
			}
			let before = Instant::now();
			'render: loop {
				// handle new events
				while let Ok(event) = me.input_receiver.try_recv() {
					log::trace!("Input event {:?}", event);
					match event {
						InputEvent::Exit => break 'render,
						InputEvent::WindowSize(size) => {
							me.surface_size = size;
						}
					}
				}

				// draw
				log::trace!("Draw frame before");
				me.last_before = Some(dirty_mark::before());

				render_loop_fn(&mut me, &mut t);

				if let Some(before_middle) = me.last_before_middle.take() {
					unsafe {
						dirty_mark::after(before_middle);
					}
				}
				log::trace!("Draw frame after\n");
			}
			let after = Instant::now().duration_since(before);
			unsafe {
				dirty_mark::flush();
			}
			println!("RENDER_LOOP: {}", after.as_nanos());

			let before = Instant::now();
			cleanup_fn(&mut me, t);
			let after = Instant::now().duration_since(before);
			println!("CLEANUP: {}", after.as_nanos());
		})
	}

	fn new(
		window: Window,
		input_receiver: mpsc::Receiver<InputEvent>,
		oneoff_sender: mpsc::Sender<Window>
	) -> Self {
		let surface_size = [
			NonZeroU32::new(window.inner_size().width).unwrap(),
			NonZeroU32::new(window.inner_size().height).unwrap()
		];

		let (base, surface) = Self::setup_base(
			vulkayes_window::winit::required_extensions(&window)
				.as_ref()
				.iter()
				.copied(),
			|instance| {
				let surface =
					vulkayes_window::winit::create_surface(instance, &window, Default::default())
						.expect("Could not create surface");
				oneoff_sender.send(window).expect("could not send window");

				surface
			}
		);
		let device_memory_allocator = NaiveDeviceMemoryAllocator::new(base.device.clone());

		let swapchain = Self::setup_swapchain(surface, &base.present_queue, surface_size);

		let mut command = Self::setup_commands(&base.present_queue);

		let depth_image_view = Self::create_depth_image_view(
			&base.present_queue,
			swapchain.create_info.image_info.image_size.into(),
			&device_memory_allocator,
			&mut command
		);

		ApplicationState {
			base,
			device_memory_allocator,

			swapchain,
			surface_size,
			depth_image_view,
			command,
			input_receiver,

			last_before: None,
			last_before_middle: None
		}
	}

	/// Returns true if the swapchain was recreated last frame.
	pub const fn swapchain_recreated(&self) -> bool {
		self.swapchain.was_recreated
	}
}
