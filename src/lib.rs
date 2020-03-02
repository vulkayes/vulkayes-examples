use std::{
	cell::RefCell,
	default::Default,
	num::NonZeroU32,
	ops::{Deref, Drop}
};

use ash::{version::DeviceV1_0, vk};
use vulkayes_core::{
	ash,
	device::Device,
	entry::Entry,
	instance::{ApplicationInfo, Instance},
	physical_device::enumerate::PhysicalDeviceMemoryProperties,
	queue::Queue,
	resource::ImageSize,
	swapchain::{Swapchain, SwapchainCreateImageInfo, SwapchainImage},
	util::{fmt::VkVersion, SharingMode},
	Vrc
};
use vulkayes_window::winit::winit;
use winit::event_loop::EventLoop;

// Simple offset_of macro akin to C++ offsetof
#[macro_export]
macro_rules! offset_of {
	($base:path, $field:ident) => {{
		#[allow(unused_unsafe)]
		unsafe {
			let b: $base = mem::zeroed();
			(&b.$field as *const _ as isize) - (&b as *const _ as isize)
			}
		}};
}

pub fn record_submit_commandbuffer<D: DeviceV1_0, F: FnOnce(&D, vk::CommandBuffer)>(
	device: &D,
	command_buffer: vk::CommandBuffer,
	submit_queue: vk::Queue,
	wait_mask: &[vk::PipelineStageFlags],
	wait_semaphores: &[vk::Semaphore],
	signal_semaphores: &[vk::Semaphore],
	f: F
) {
	unsafe {
		device
			.reset_command_buffer(
				command_buffer,
				vk::CommandBufferResetFlags::RELEASE_RESOURCES
			)
			.expect("Reset command buffer failed.");

		let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
			.flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

		device
			.begin_command_buffer(command_buffer, &command_buffer_begin_info)
			.expect("Begin commandbuffer");
		f(device, command_buffer);
		device
			.end_command_buffer(command_buffer)
			.expect("End commandbuffer");

		let submit_fence = device
			.create_fence(&vk::FenceCreateInfo::default(), None)
			.expect("Create fence failed.");

		let command_buffers = vec![command_buffer];

		let submit_info = vk::SubmitInfo::builder()
			.wait_semaphores(wait_semaphores)
			.wait_dst_stage_mask(wait_mask)
			.command_buffers(&command_buffers)
			.signal_semaphores(signal_semaphores);

		device
			.queue_submit(submit_queue, &[submit_info.build()], submit_fence)
			.expect("queue submit failed.");
		device
			.wait_for_fences(&[submit_fence], true, std::u64::MAX)
			.expect("Wait for fence failed.");
		device.destroy_fence(submit_fence, None);
	}
}

pub fn find_memorytype_index(
	memory_req: &vk::MemoryRequirements,
	memory_prop: &PhysicalDeviceMemoryProperties,
	flags: vk::MemoryPropertyFlags
) -> Option<u32> {
	// Try to find an exactly matching memory flag
	let best_suitable_index =
		find_memorytype_index_f(memory_req, memory_prop, flags, |property_flags, flags| {
			property_flags == flags
		});
	if best_suitable_index.is_some() {
		return best_suitable_index
	}
	// Otherwise find a memory flag that works
	find_memorytype_index_f(memory_req, memory_prop, flags, |property_flags, flags| {
		property_flags & flags == flags
	})
}

pub fn find_memorytype_index_f<F: Fn(vk::MemoryPropertyFlags, vk::MemoryPropertyFlags) -> bool>(
	memory_req: &vk::MemoryRequirements,
	memory_prop: &PhysicalDeviceMemoryProperties,
	flags: vk::MemoryPropertyFlags,
	f: F
) -> Option<u32> {
	let mut memory_type_bits = memory_req.memory_type_bits;
	for (index, ref memory_type) in memory_prop.memory_types.iter().enumerate() {
		if memory_type_bits & 1 == 1 && f(memory_type.property_flags, flags) {
			return Some(index as u32)
		}
		memory_type_bits >>= 1;
	}
	None
}

pub struct ExampleBase {
	pub instance: Vrc<Instance>, //
	pub device: Vrc<Device>,     //

	pub event_loop: RefCell<EventLoop<()>>, //

	pub device_memory_properties: PhysicalDeviceMemoryProperties, //
	pub present_queue: Vrc<Queue>,                                //

	pub window: winit::window::Window, //
	pub surface_size: ImageSize,

	pub swapchain: Vrc<Swapchain>,                //
	pub present_images: Vec<Vrc<SwapchainImage>>, //
	pub present_image_views: Vec<vk::ImageView>,

	pub pool: vk::CommandPool,
	pub draw_command_buffer: vk::CommandBuffer,
	pub setup_command_buffer: vk::CommandBuffer,

	pub depth_image: vk::Image,
	pub depth_image_view: vk::ImageView,
	pub depth_image_memory: vk::DeviceMemory,

	pub present_complete_semaphore: vk::Semaphore,
	pub rendering_complete_semaphore: vk::Semaphore
}

impl ExampleBase {
	pub fn render_loop<F: Fn()>(&self, f: F) {
		use winit::{
			event::{Event, WindowEvent},
			platform::desktop::EventLoopExtDesktop
		};

		self.event_loop
			.borrow_mut()
			.run_return(|event, _target, control_flow| {
				// Run update
				f();

				// Handle window close event
				match event {
					Event::WindowEvent {
						event: WindowEvent::CloseRequested,
						..
					} => {
						*control_flow = winit::event_loop::ControlFlow::Exit;
					}
					_ => ()
				}
			});
	}

	pub fn new(window_width: u32, window_height: u32) -> Self {
		// Register a logger since Vulkayes logs through the log crate
		let logger = edwardium_logger::Logger::new(
			[edwardium_logger::targets::stderr::StderrTarget::new(
				log::Level::Trace
			)],
			std::time::Instant::now()
		);
		logger.init_boxed().expect("Could not initialize logger");

		let event_loop = EventLoop::new();
		let window = winit::window::WindowBuilder::new()
			.with_title("Ash - Example")
			.with_inner_size(winit::dpi::LogicalSize::new(
				f64::from(window_width),
				f64::from(window_height)
			))
			.build(&event_loop)
			.unwrap();

		// Create Entry, which is the entry point into the Vulkan API.
		// The default entry is loaded as a dynamic library.
		let entry = Entry::new().unwrap();

		// Create instance from loaded entry
		// Also enable validation layers and require surface extensions as defined in vulkayes_window
		// Lastly, register Default debug callbacks that log using the log crate
		let instance = Instance::new(
			entry,
			ApplicationInfo {
				application_name: "VulkanTriangle",
				engine_name: "VulkanTriangle",
				api_version: VkVersion::new(1, 0, 0),
				..Default::default()
			},
			["VK_LAYER_LUNARG_standard_validation"].iter().map(|&s| s),
			vulkayes_window::winit::required_surface_extensions()
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
			vulkayes_window::winit::create_surface(instance.clone(), &window, Default::default())
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
					flags: Default::default(),
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

		let (swapchain, present_images) = {
			let surface_format = surface
				.physical_device_surface_formats(device.physical_device())
				.unwrap()
				.drain(..)
				.nth(0)
				.expect("Unable to find suitable surface format.");
			let surface_capabilities = surface
				.physical_device_surface_capabilities(device.physical_device())
				.unwrap();

			let desired_image_count = match surface_capabilities.max_image_count {
				0 => surface_capabilities.min_image_count + 1,
				a => (surface_capabilities.min_image_count + 1).min(a)
			};
			let surface_resolution = match surface_capabilities.current_extent.width {
				std::u32::MAX => [
					NonZeroU32::new(window_width).unwrap(),
					NonZeroU32::new(window_height).unwrap()
				],
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
			let present_mode = surface
				.physical_device_surface_present_modes(device.physical_device())
				.unwrap()
				.into_iter()
				.find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
				.unwrap_or(vk::PresentModeKHR::FIFO);

			let swapchain_data = Swapchain::new(
				device.clone(),
				surface,
				SwapchainCreateImageInfo {
					min_image_count: NonZeroU32::new(desired_image_count).unwrap(),
					image_format: surface_format.format,
					image_color_space: surface_format.color_space,
					image_extent: surface_resolution,
					image_array_layers: vulkayes_core::NONZEROU32_ONE,
					image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
				},
				SharingMode::<[u32; 0]>::Exclusive,
				pre_transform,
				vk::CompositeAlphaFlagsKHR::OPAQUE,
				present_mode,
				true,
				Default::default()
			)
			.expect("Could not create swapchain");

			(swapchain_data.swapchain, swapchain_data.images)
		};

		unsafe {
			let pool_create_info = vk::CommandPoolCreateInfo::builder()
				.flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
				.queue_family_index(present_queue.queue_family_index());

			let pool = device.create_command_pool(&pool_create_info, None).unwrap();

			let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
				.command_buffer_count(2)
				.command_pool(pool)
				.level(vk::CommandBufferLevel::PRIMARY);

			let command_buffers = device
				.allocate_command_buffers(&command_buffer_allocate_info)
				.unwrap();
			let setup_command_buffer = command_buffers[0];
			let draw_command_buffer = command_buffers[1];

			let present_image_views: Vec<vk::ImageView> = present_images
				.iter()
				.map(|image| {
					let create_view_info = vk::ImageViewCreateInfo::builder()
						.view_type(vk::ImageViewType::TYPE_2D)
						.format(image.format())
						.components(vk::ComponentMapping {
							r: vk::ComponentSwizzle::R,
							g: vk::ComponentSwizzle::G,
							b: vk::ComponentSwizzle::B,
							a: vk::ComponentSwizzle::A
						})
						.subresource_range(vk::ImageSubresourceRange {
							aspect_mask: vk::ImageAspectFlags::COLOR,
							base_mip_level: 0,
							level_count: 1,
							base_array_layer: 0,
							layer_count: 1
						})
						.image(*image.deref().deref().deref());
					device.create_image_view(&create_view_info, None).unwrap()
				})
				.collect();

			let device_memory_properties = device.physical_device().memory_properties();
			let depth_image_create_info = vk::ImageCreateInfo::builder()
				.image_type(vk::ImageType::TYPE_2D)
				.format(vk::Format::D16_UNORM)
				.extent(present_images[0].size().into())
				.mip_levels(1)
				.array_layers(1)
				.samples(vk::SampleCountFlags::TYPE_1)
				.tiling(vk::ImageTiling::OPTIMAL)
				.usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
				.sharing_mode(vk::SharingMode::EXCLUSIVE);

			let depth_image = device.create_image(&depth_image_create_info, None).unwrap();
			let depth_image_memory_req = device.get_image_memory_requirements(depth_image);
			let depth_image_memory_index = find_memorytype_index(
				&depth_image_memory_req,
				&device_memory_properties,
				vk::MemoryPropertyFlags::DEVICE_LOCAL
			)
			.expect("Unable to find suitable memory index for depth image.");

			let depth_image_allocate_info = vk::MemoryAllocateInfo::builder()
				.allocation_size(depth_image_memory_req.size)
				.memory_type_index(depth_image_memory_index);

			let depth_image_memory = device
				.allocate_memory(&depth_image_allocate_info, None)
				.unwrap();

			device
				.bind_image_memory(depth_image, depth_image_memory, 0)
				.expect("Unable to bind depth image memory");

			record_submit_commandbuffer(
				device.deref().deref(),
				setup_command_buffer,
				*present_queue.deref().deref(),
				&[],
				&[],
				&[],
				|device, setup_command_buffer| {
					let layout_transition_barriers = vk::ImageMemoryBarrier::builder()
						.image(depth_image)
						.dst_access_mask(
							vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
								| vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
						)
						.new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
						.old_layout(vk::ImageLayout::UNDEFINED)
						.subresource_range(
							vk::ImageSubresourceRange::builder()
								.aspect_mask(vk::ImageAspectFlags::DEPTH)
								.layer_count(1)
								.level_count(1)
								.build()
						);

					device.cmd_pipeline_barrier(
						setup_command_buffer,
						vk::PipelineStageFlags::BOTTOM_OF_PIPE,
						vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
						vk::DependencyFlags::empty(),
						&[],
						&[],
						&[layout_transition_barriers.build()]
					);
				}
			);

			let depth_image_view_info = vk::ImageViewCreateInfo::builder()
				.subresource_range(
					vk::ImageSubresourceRange::builder()
						.aspect_mask(vk::ImageAspectFlags::DEPTH)
						.level_count(1)
						.layer_count(1)
						.build()
				)
				.image(depth_image)
				.format(depth_image_create_info.format)
				.view_type(vk::ImageViewType::TYPE_2D);

			let depth_image_view = device
				.create_image_view(&depth_image_view_info, None)
				.unwrap();

			let semaphore_create_info = vk::SemaphoreCreateInfo::default();

			let present_complete_semaphore = device
				.create_semaphore(&semaphore_create_info, None)
				.unwrap();
			let rendering_complete_semaphore = device
				.create_semaphore(&semaphore_create_info, None)
				.unwrap();
			ExampleBase {
				event_loop: RefCell::new(event_loop),
				instance,
				device,
				device_memory_properties,
				present_queue,
				swapchain,
				surface_size: present_images[0].size(),
				present_images,
				present_image_views,
				pool,
				draw_command_buffer,
				setup_command_buffer,
				depth_image,
				depth_image_view,
				present_complete_semaphore,
				rendering_complete_semaphore,
				window,
				depth_image_memory
			}
		}
	}
}

impl Drop for ExampleBase {
	fn drop(&mut self) {
		unsafe {
			self.device.device_wait_idle().unwrap();

			self.device
				.destroy_semaphore(self.present_complete_semaphore, None);
			self.device
				.destroy_semaphore(self.rendering_complete_semaphore, None);

			self.device.destroy_image_view(self.depth_image_view, None);
			self.device.destroy_image(self.depth_image, None);
			self.device.free_memory(self.depth_image_memory, None);

			for &image_view in self.present_image_views.iter() {
				self.device.destroy_image_view(image_view, None);
			}

			self.device.destroy_command_pool(self.pool, None);
		}
	}
}
