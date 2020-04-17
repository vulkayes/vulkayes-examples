use std::{default::Default, num::NonZeroU32, ops::Deref, sync::mpsc};

// use winit::window::Window;

use ash::{version::DeviceV1_0, vk};

use vulkayes_core::{
	ash,
	command::{buffer::CommandBuffer, pool::CommandPool},
	device::Device,
	entry::Entry,
	instance::{ApplicationInfo, Instance},
	log,
	memory::device::naive::NaiveDeviceMemoryAllocator,
	queue::Queue,
	resource::image::{
		params::{AllocatorParams, ImageSize, ImageViewRange, MipmapLevels},
		view::ImageView,
		Image
	},
	swapchain::{image::SwapchainCreateImageInfo, Swapchain, SwapchainCreateInfo},
	sync::semaphore::{BinarySemaphore, Semaphore},
	util::fmt::VkVersion,
	Vrc
};

// use vulkayes_window::winit::winit;
// use winit::event_loop::EventLoop;

use minifb::Window;
use vulkayes_window::minifb::minifb;

#[macro_use]
pub mod dirty_mark;

pub mod state;

pub fn record_command_buffer(
	command_buffer: &Vrc<CommandBuffer>,
	f: impl FnOnce(&Vrc<CommandBuffer>)
) {
	unsafe {
		{
			let cb_lock = command_buffer.lock().expect("vutex poisoned");
			// TODO: This only works because we wait for the fence at the end of each submit.
			// In real-life applications this isn't viable
			command_buffer
				.pool()
				.device()
				.reset_command_buffer(*cb_lock, vk::CommandBufferResetFlags::RELEASE_RESOURCES)
				.expect("Reset command buffer failed.");

			let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
				.flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

			command_buffer
				.pool()
				.device()
				.begin_command_buffer(*cb_lock, &command_buffer_begin_info)
				.expect("Begin commandbuffer");
		}

		f(command_buffer);

		{
			let cb_lock = command_buffer.lock().expect("vutex poisoned");
			command_buffer
				.pool()
				.device()
				.end_command_buffer(*cb_lock)
				.expect("End commandbuffer");
		}
	}
}
pub fn submit_command_buffer_simple(
	command_buffer: &Vrc<CommandBuffer>,
	submit_queue: &Vrc<Queue>
) {
	vulkayes_core::const_queue_submit! {
		pub fn submit(
			&queue,
			waits: [&Semaphore; 0],
			stages: [vk::PipelineStageFlags; _],
			buffers: [&CommandBuffer; 1],
			signals: [&Semaphore; 0],
			fence: Option<&Fence>
		) -> Result<(), QueueSubmitError>;
	};

	let submit_fence = vulkayes_core::sync::fence::Fence::new(
		command_buffer.pool().device().clone(),
		false,
		Default::default()
	)
	.expect("Create fence failed.");

	submit(
		submit_queue,
		[],
		[],
		[command_buffer.deref()],
		[],
		Some(submit_fence.deref())
	)
	.expect("queue submit failed");

	submit_fence
		.wait(Default::default())
		.expect("fence wait failed");
}
pub fn submit_command_buffer(
	command_buffer: &Vrc<CommandBuffer>,
	submit_queue: &Vrc<Queue>,
	wait_semaphore: &Vrc<Semaphore>,
	wait_mask: vk::PipelineStageFlags,
	signal_semaphore: &Vrc<Semaphore>
) {
	vulkayes_core::const_queue_submit! {
		pub fn submit(
			&queue,
			waits: [&Semaphore; 1],
			stages: [vk::PipelineStageFlags; _],
			buffers: [&CommandBuffer; 1],
			signals: [&Semaphore; 1],
			fence: Option<&Fence>
		) -> Result<(), QueueSubmitError>;
	};

	let submit_fence = vulkayes_core::sync::fence::Fence::new(
		command_buffer.pool().device().clone(),
		false,
		Default::default()
	)
	.expect("Create fence failed.");

	submit(
		submit_queue,
		[wait_semaphore],
		[wait_mask],
		[command_buffer],
		[signal_semaphore],
		Some(&submit_fence)
	)
	.expect("queue submit failed");

	submit_fence
		.wait(Default::default())
		.expect("fence wait failed");
}

#[derive(Debug)]
enum InputEvent {
	Exit
}
pub struct ExampleBase {
	pub instance: Vrc<Instance>,
	pub device: Vrc<Device>,

	input_receiver: mpsc::Receiver<InputEvent>,

	pub present_queue: Vrc<Queue>,

	// pub window: winit::window::Window,
	pub surface_size: ImageSize,

	pub swapchain: Vrc<Swapchain>,
	pub swapchain_create_info: SwapchainCreateInfo<[u32; 1]>,
	pub present_image_views: Vec<Vrc<ImageView>>,

	pub command_pool: Vrc<CommandPool>,
	pub draw_command_buffer: Vrc<CommandBuffer>,
	pub setup_command_buffer: Vrc<CommandBuffer>,

	pub device_memory_allocator: NaiveDeviceMemoryAllocator,
	pub depth_image_view: Vrc<ImageView>,

	pub present_complete_semaphore: BinarySemaphore,
	pub rendering_complete_semaphore: BinarySemaphore,

	pub swapchain_outdated: u8
}
impl ExampleBase {
	/// Converts the current thread to the input thread by created an event loop in it.
	///
	/// A new thread is spawned with `new_thread_fn`.
	///
	/// This function is designed this way because on MacOS only the main thread can have an EventLoop on it,
	/// so this must be called from the main thread.
	pub fn with_input_thread(
		window_size: [NonZeroU32; 2],
		new_thread_fn: impl FnOnce(Self) + Send + 'static
	) {
		// Register a logger since Vulkayes logs through the log crate
		let logger = edwardium_logger::Logger::new(
			[edwardium_logger::targets::stderr::StderrTarget::new(
				log::Level::Trace
			)],
			std::time::Instant::now()
		);
		logger.init_boxed().expect("Could not initialize logger");

		// let event_loop = EventLoop::new();
		// let window = winit::window::WindowBuilder::new()
		// 	.with_title("Ash - Example")
		// 	.with_inner_size(winit::dpi::LogicalSize::new(
		// 		f64::from(window_size[0].get()),
		// 		f64::from(window_size[1].get())
		// 	))
		// 	.build(&event_loop)
		// 	.expect("could not create window");
		let mut window = Window::new(
			"Ash - Example",
			window_size[0].get() as usize,
			window_size[1].get() as usize,
			Default::default()
		)
		.expect("could not create window");
		window.update();

		struct Windowino(Window);
		unsafe impl Send for Windowino {};
		unsafe impl Sync for Windowino {};
		let send_me = Windowino(window);

		// A little something about macOS and winit
		// On macOS, both the even loop and the window must stay in the main thread while rendering.
		// However, the window is needed to create surface, which needs an instance, which means it needs to happen in the renderer thread.
		// Here we have to move the window into the new thread and then send it back using a channel.
		let (oneoff_sender, oneoff_receiver) = mpsc::channel::<Windowino>();

		let (input_sender, input_receiver) = mpsc::channel::<InputEvent>();
		std::thread::spawn(move || {
			let w = send_me;
			let window = w.0;

			let base = Self::new(window_size, &window, input_receiver);

			let send_me = Windowino(window);
			oneoff_sender.send(send_me).expect("could not send window");

			new_thread_fn(base);
		});

		// We don't actually need the window after, but it has to be in this thread. Weird.
		let mut window = oneoff_receiver
			.recv()
			.expect("could not receive window back")
			.0;

		// Technically result is now `!`, but this can be trivially replaced by `run_return` where supported
		// event_loop.run(move |event, _target, control_flow| {
		// 	// Handle window close event
		// 	match event {
		// 		winit::event::Event::WindowEvent {
		// 			event: winit::event::WindowEvent::CloseRequested,
		// 			..
		// 		} => {
		// 			input_sender
		// 				.send(InputEvent::Exit)
		// 				.expect("could not send input event");
		// 			*control_flow = winit::event_loop::ControlFlow::Exit;
		// 		}
		// 		_ => ()
		// 	}
		// })

		while window.is_open() {
			window.update();
		}
	}

	fn draw(&mut self, f: &impl Fn(&mut Self, u32)) {
		vulkayes_core::const_queue_present!(
			pub fn queue_present(
				&queue,
				waits: [&Semaphore; 1],
				images: [&SwapchainImage; 1],
				result_for_all: bool
			) -> QueuePresentMultipleResult<[QueuePresentResult; _]>;
		);

		if self.swapchain_outdated > 120 {
			panic!("Swapchain outdated for 120 frames");
		} else if self.swapchain_outdated > 0 {
			// let _window_size = self.window.inner_size();
			// TODO: Recreate swapchain and framebuffers

			if self.swapchain_outdated > 1 {
				log::warn!(
					"Swapchain outdated for {} frames now",
					self.swapchain_outdated
				);
			}

			// self.swapchain_outdated = 0;
		}

		let present_index = match self.swapchain.acquire_next(
			Default::default(),
			(&self.present_complete_semaphore).into()
		) {
			Ok(vulkayes_core::swapchain::error::AcquireResultValue::SUCCESS(i)) => i,
			Ok(vulkayes_core::swapchain::error::AcquireResultValue::SUBOPTIMAL_KHR(_))
			| Err(vulkayes_core::swapchain::error::AcquireError::ERROR_OUT_OF_DATE_KHR) => {
				self.swapchain_outdated += 1;
				return
			}
			Err(e) => panic!("{}", e)
		};

		f(self, present_index);

		match queue_present(
			&self.present_queue,
			[&self.rendering_complete_semaphore],
			[self.present_image_views[present_index as usize]
				.image()
				.try_swapchain_image()
				.unwrap()],
			false
		) {
			vulkayes_core::queue::error::QueuePresentMultipleResult::Single(result) => match result
			{
				Ok(vulkayes_core::queue::error::QueuePresentResultValue::SUCCESS) => (),
				Ok(vulkayes_core::queue::error::QueuePresentResultValue::SUBOPTIMAL_KHR)
				| Err(vulkayes_core::queue::error::QueuePresentError::ERROR_OUT_OF_DATE_KHR) => {
					self.swapchain_outdated += 1;
				}
				Err(e) => panic!("{}", e)
			},
			_ => unreachable!()
		}
	}

	pub fn render_loop(&mut self, f: impl Fn(&mut Self, u32)) {
		unsafe {
			dirty_mark::init();
		}

		'render: loop {
			// handle new events
			while let Ok(event) = self.input_receiver.try_recv() {
				log::trace!("Input event {:?}", event);
				match event {
					InputEvent::Exit => break 'render
				}
			}

			// draw
			log::trace!("Draw frame before");
			let before = dirty_mark::before();
			self.draw(&f);
			unsafe {
				dirty_mark::after([before, before]);
			}
			log::trace!("Draw frame after\n");
		}

		unsafe {
			dirty_mark::flush();
		}
	}

	fn new(
		window_size: [NonZeroU32; 2],
		window: &Window,
		input_receiver: mpsc::Receiver<InputEvent>
	) -> Self {
		// Create Entry, which is the entry point into the Vulkan API.
		// The default entry is loaded as a dynamic library.
		let entry = Entry::new().unwrap();

		// Create instance from loaded entry
		// Also enable validation layers and require surface extensions as defined in vulkayes_window
		let layers = sub_release! {
			["VK_LAYER_LUNARG_standard_validation"].iter().map(|&s| s),
			std::iter::empty()
		};
		// Lastly, register Default debug callbacks that log using the log crate
		let instance = Instance::new(
			entry,
			ApplicationInfo {
				application_name: "VulkanTriangle",
				engine_name: "VulkanTriangle",
				api_version: VkVersion::new(1, 0, 0),
				..Default::default()
			},
			layers,
			vulkayes_window::minifb::required_extensions(&window)
				.as_ref()
				.iter()
				.map(|&s| s)
				.chain(std::iter::once(
					ash::extensions::ext::DebugReport::name().to_str().unwrap()
				)),
			Default::default(),
			vulkayes_core::instance::debug::DebugCallback::Default()
		)
		.expect("Could not create instance");

		// Create a surface using the vulkayes_window::winit feature
		let surface =
			vulkayes_window::minifb::create_surface(instance.clone(), &window, Default::default())
				.expect("Could not create surface");

		let (device, present_queue) = {
			let (physical_device, queue_family_index) = instance
				.physical_devices()
				.expect("Physical device enumeration error")
				.into_iter()
				.filter_map(|physical_device| {
					for (queue_family_index, info) in physical_device
						.queue_family_properties()
						.into_iter()
						.enumerate()
					{
						let supports_graphics = info.queue_flags.contains(vk::QueueFlags::GRAPHICS);
						let supports_surface = surface
							.physical_device_surface_support(
								&physical_device,
								queue_family_index as u32
							)
							.unwrap();

						if supports_graphics && supports_surface {
							return Some((physical_device, queue_family_index))
						}
					}

					None
				})
				.nth(0)
				.expect("Couldn't find suitable device.");

			let mut device_data = Device::new(
				physical_device,
				[vulkayes_core::device::QueueCreateInfo {
					queue_family_index: queue_family_index as u32,
					queue_priorities: [1.0]
				}],
				None,
				[ash::extensions::khr::Swapchain::name().to_str().unwrap()]
					.iter()
					.map(|&s| s),
				vk::PhysicalDeviceFeatures {
					shader_clip_distance: 1,
					..Default::default()
				},
				Default::default()
			)
			.expect("Could not create device");

			let present_queue = device_data.queues.drain(..).nth(0).unwrap();

			(device_data.device, present_queue)
		};

		let device_memory_allocator = NaiveDeviceMemoryAllocator::new(device.clone());

		let swapchain_create_info =
			Self::swapchain_create_info(&surface, &present_queue, window_size);
		let surface_size: ImageSize = swapchain_create_info.image_info.image_size.into();
		let (swapchain, present_images) = {
			let s = Swapchain::new(
				device.clone(),
				surface,
				swapchain_create_info,
				Default::default()
			)
			.expect("could not create swapchain");
			(s.swapchain, s.images)
		};

		let command_pool = CommandPool::new(
			&present_queue,
			vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
			Default::default()
		)
		.expect("Could not create command pool");

		let mut command_buffers = CommandBuffer::new_multiple(
			command_pool.clone(),
			true,
			std::num::NonZeroU32::new(2).unwrap()
		)
		.expect("Could not allocate command buffers");

		let draw_command_buffer = command_buffers.pop().unwrap();
		let setup_command_buffer = command_buffers.pop().unwrap();

		let present_image_views: Vec<Vrc<ImageView>> = present_images
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
			.collect();

		let depth_image_view = {
			let depth_image = Image::new(
				device.clone(),
				vk::Format::D16_UNORM,
				surface_size.into(),
				Default::default(),
				vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
				present_queue.deref().into(),
				AllocatorParams::Some {
					allocator: &device_memory_allocator,
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

		record_command_buffer(&setup_command_buffer, |command_buffer| {
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
		submit_command_buffer_simple(&setup_command_buffer, &present_queue);

		let present_complete_semaphore = Semaphore::binary(device.clone(), Default::default())
			.expect("Could not create semaphore");
		let rendering_complete_semaphore = Semaphore::binary(device.clone(), Default::default())
			.expect("Could not create semaphore");

		ExampleBase {
			input_receiver,

			instance,
			device,
			present_queue,
			swapchain,
			swapchain_create_info,
			surface_size,
			present_image_views,
			command_pool,
			draw_command_buffer,
			setup_command_buffer,
			device_memory_allocator,
			depth_image_view,
			present_complete_semaphore,
			rendering_complete_semaphore,
			swapchain_outdated: 0
		}
	}

	pub fn swapchain_create_info(
		surface: &vulkayes_core::surface::Surface,
		queue: &Queue,
		window_size: [NonZeroU32; 2]
	) -> SwapchainCreateInfo<[u32; 1]> {
		let surface_format = surface
			.physical_device_surface_formats(queue.device().physical_device())
			.unwrap()
			.drain(..)
			.nth(0)
			.expect("Unable to find suitable surface format.");
		let surface_capabilities = surface
			.physical_device_surface_capabilities(queue.device().physical_device())
			.unwrap();

		let desired_image_count = match surface_capabilities.max_image_count {
			0 => surface_capabilities.min_image_count + 1,
			a => (surface_capabilities.min_image_count + 1).min(a)
		};
		let surface_resolution = match surface_capabilities.current_extent.width {
			std::u32::MAX => window_size,
			_ => [
				NonZeroU32::new(surface_capabilities.current_extent.width).unwrap(),
				NonZeroU32::new(surface_capabilities.current_extent.height).unwrap()
			]
		};
		let pre_transform = if surface_capabilities
			.supported_transforms
			.contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
		{
			vk::SurfaceTransformFlagsKHR::IDENTITY
		} else {
			surface_capabilities.current_transform
		};
		let present_mode = sub_release! {
			surface
			.physical_device_surface_present_modes(queue.device().physical_device()).unwrap()
			.into_iter().find(|&mode| mode == vk::PresentModeKHR::MAILBOX).unwrap_or(vk::PresentModeKHR::FIFO),
			vk::PresentModeKHR::IMMEDIATE
		};

		SwapchainCreateInfo {
			image_info: SwapchainCreateImageInfo {
				min_image_count: NonZeroU32::new(desired_image_count).unwrap(),
				image_format: surface_format.format,
				image_color_space: surface_format.color_space,
				image_size: ImageSize::new_2d(
					surface_resolution[0],
					surface_resolution[1],
					vulkayes_core::NONZEROU32_ONE,
					MipmapLevels::One()
				),
				image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
			},
			sharing_mode: queue.into(),
			pre_transform,
			composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
			present_mode,
			clipped: true
		}
	}
}
