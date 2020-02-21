use std::{
	cell::RefCell,
	default::Default,
	ops::{Deref, Drop}
};

use ash::{
	extensions::{
		khr::{Surface, Swapchain}
	},
	version::{DeviceV1_0},
	vk
};

use vulkayes_core::{
	ash,
	device::Device,
	entry::Entry,
	instance::{ApplicationInfo, Instance},
	physical_device::enumerate::PhysicalDeviceMemoryProperties,
	queue::Queue,
	util::fmt::VkVersion,
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
	device: &D, command_buffer: vk::CommandBuffer, submit_queue: vk::Queue,
	wait_mask: &[vk::PipelineStageFlags], wait_semaphores: &[vk::Semaphore],
	signal_semaphores: &[vk::Semaphore], f: F
) {
	unsafe {
		device
			.reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::RELEASE_RESOURCES)
			.expect("Reset command buffer failed.");

		let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
			.flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

		device
			.begin_command_buffer(command_buffer, &command_buffer_begin_info)
			.expect("Begin commandbuffer");
		f(device, command_buffer);
		device.end_command_buffer(command_buffer).expect("End commandbuffer");

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
	memory_req: &vk::MemoryRequirements, memory_prop: &PhysicalDeviceMemoryProperties,
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
	memory_req: &vk::MemoryRequirements, memory_prop: &PhysicalDeviceMemoryProperties,
	flags: vk::MemoryPropertyFlags, f: F
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

	pub swapchain_loader: Swapchain,
	pub events_loop: RefCell<EventLoop<()>>,

	pub device_memory_properties: PhysicalDeviceMemoryProperties, //
	pub present_queue: Vrc<Queue>,                                //

	pub surface: vulkayes_window::winit::Surface, //
	pub surface_format: vk::SurfaceFormatKHR,
	pub surface_resolution: vk::Extent2D,

	pub swapchain: vk::SwapchainKHR,
	pub present_images: Vec<vk::Image>,
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
		use winit::platform::desktop::EventLoopExtDesktop;
		use winit::event::{Event, WindowEvent};

		self.events_loop.borrow_mut().run_return(|event, _target, control_flow| {
			// Run update
			f();

			// Handle window close event
			match event {
				Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
					*control_flow = winit::event_loop::ControlFlow::Exit;
				}
				_ => ()
			}
		});
	}

	pub fn new(window_width: u32, window_height: u32) -> Self {
		let logger = edwardium_logger::Logger::new(
			[edwardium_logger::targets::stderr::StderrTarget::new(log::Level::Trace)],
			std::time::Instant::now()
		);
		logger.init_boxed().expect("Could not initialize logger");

		unsafe {
			let events_loop =  EventLoop::new();
			let window = winit::window::WindowBuilder::new()
				.with_title("Ash - Example")
				.with_inner_size(winit::dpi::LogicalSize::new(
					f64::from(window_width),
					f64::from(window_height)
				))
				.build(&events_loop)
				.unwrap();

			let entry = Entry::new().unwrap();

			let instance = Instance::new(
				entry,
				ApplicationInfo {
					application_name: "VulkanTriangle",
					engine_name: "VulkanTriangle",
					api_version: VkVersion::new(1, 0, 0),
					..Default::default()
				},
				["VK_LAYER_LUNARG_standard_validation"].iter().map(|&s| s),
				vulkayes_window::winit::required_surface_extensions().as_ref()
					.iter().map(|&s| s)
					.chain(
						std::iter::once(
							ash::extensions::ext::DebugReport::name().to_str().unwrap()
						)
					),
				Default::default(),
				vulkayes_core::instance::debug::DebugCallback::Default()
			)
			.expect("Could not create instance");

			let surface = vulkayes_window::winit::create_surface(
				instance.clone(),
				window,
				Default::default()
			).expect("Could not create surface");

			let pdevices = instance.physical_devices().expect("Physical device enumeration error");

			let surface_loader = Surface::new(instance.entry().deref(), instance.deref().deref());

			let (pdevice, queue_family_index) = pdevices
				.filter_map(|pdevice| {
					pdevice
						.queue_family_properties()
						.iter()
						.enumerate()
						.filter_map(|(index, ref info)| {
							let supports_graphic_and_surface =
								info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
									&& surface_loader.get_physical_device_surface_support(
										*pdevice,
										index as u32,
										*surface
									);

							if supports_graphic_and_surface {
								Some((pdevice.clone(), index))
							} else {
								None
							}
						})
						.nth(0)
				})
				.nth(0)
				.expect("Couldn't find suitable device.");

			let (device, mut queues) = Device::new(
				instance.clone(),
				[vulkayes_core::device::QueueCreateInfo {
					queue_family_index: queue_family_index as u32,
					flags: Default::default(),
					queue_priorities: [1.0]
				}],
				None,
				[Swapchain::name().to_str().unwrap()].iter().map(|&s| s),
				vk::PhysicalDeviceFeatures { shader_clip_distance: 1, ..Default::default() },
				pdevice,
				Default::default()
			)
			.expect("Could not create device");

			let present_queue = queues.remove(0);

			let surface_formats = surface_loader
				.get_physical_device_surface_formats(*device.physical_device().deref(), *surface)
				.unwrap();
			let surface_format = surface_formats
				.iter()
				.map(|sfmt| match sfmt.format {
					vk::Format::UNDEFINED => vk::SurfaceFormatKHR {
						format: vk::Format::B8G8R8_UNORM,
						color_space: sfmt.color_space
					},
					_ => *sfmt
				})
				.nth(0)
				.expect("Unable to find suitable surface format.");
			let surface_capabilities = surface_loader
				.get_physical_device_surface_capabilities(
					*device.physical_device().deref(),
					*surface
				)
				.unwrap();
			let mut desired_image_count = surface_capabilities.min_image_count + 1;
			if surface_capabilities.max_image_count > 0
				&& desired_image_count > surface_capabilities.max_image_count
			{
				desired_image_count = surface_capabilities.max_image_count;
			}
			let surface_resolution = match surface_capabilities.current_extent.width {
				std::u32::MAX => vk::Extent2D { width: window_width, height: window_height },
				_ => surface_capabilities.current_extent
			};
			let pre_transform = if surface_capabilities
				.supported_transforms
				.contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
			{
				vk::SurfaceTransformFlagsKHR::IDENTITY
			} else {
				surface_capabilities.current_transform
			};
			let present_modes = surface_loader
				.get_physical_device_surface_present_modes(
					*device.physical_device().deref(),
					*surface
				)
				.unwrap();
			let present_mode = present_modes
				.iter()
				.cloned()
				.find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
				.unwrap_or(vk::PresentModeKHR::FIFO);
			let swapchain_loader = Swapchain::new(instance.deref().deref(), device.deref().deref());

			let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
				.surface(*surface)
				.min_image_count(desired_image_count)
				.image_color_space(surface_format.color_space)
				.image_format(surface_format.format)
				.image_extent(surface_resolution)
				.image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
				.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
				.pre_transform(pre_transform)
				.composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
				.present_mode(present_mode)
				.clipped(true)
				.image_array_layers(1);

			let swapchain =
				swapchain_loader.create_swapchain(&swapchain_create_info, None).unwrap();

			let pool_create_info = vk::CommandPoolCreateInfo::builder()
				.flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
				.queue_family_index(present_queue.queue_family_index());

			let pool = device.create_command_pool(&pool_create_info, None).unwrap();

			let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
				.command_buffer_count(2)
				.command_pool(pool)
				.level(vk::CommandBufferLevel::PRIMARY);

			let command_buffers =
				device.allocate_command_buffers(&command_buffer_allocate_info).unwrap();
			let setup_command_buffer = command_buffers[0];
			let draw_command_buffer = command_buffers[1];

			let present_images = swapchain_loader.get_swapchain_images(swapchain).unwrap();
			let present_image_views: Vec<vk::ImageView> = present_images
				.iter()
				.map(|&image| {
					let create_view_info = vk::ImageViewCreateInfo::builder()
						.view_type(vk::ImageViewType::TYPE_2D)
						.format(surface_format.format)
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
						.image(image);
					device.create_image_view(&create_view_info, None).unwrap()
				})
				.collect();

			let device_memory_properties = device.physical_device().memory_properties();
			let depth_image_create_info = vk::ImageCreateInfo::builder()
				.image_type(vk::ImageType::TYPE_2D)
				.format(vk::Format::D16_UNORM)
				.extent(vk::Extent3D {
					width: surface_resolution.width,
					height: surface_resolution.height,
					depth: 1
				})
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

			let depth_image_memory =
				device.allocate_memory(&depth_image_allocate_info, None).unwrap();

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

			let depth_image_view = device.create_image_view(&depth_image_view_info, None).unwrap();

			let semaphore_create_info = vk::SemaphoreCreateInfo::default();

			let present_complete_semaphore =
				device.create_semaphore(&semaphore_create_info, None).unwrap();
			let rendering_complete_semaphore =
				device.create_semaphore(&semaphore_create_info, None).unwrap();
			ExampleBase {
				events_loop: RefCell::new(events_loop),
				instance,
				device,
				device_memory_properties,
				surface_format,
				present_queue,
				surface_resolution,
				swapchain_loader,
				swapchain,
				present_images,
				present_image_views,
				pool,
				draw_command_buffer,
				setup_command_buffer,
				depth_image,
				depth_image_view,
				present_complete_semaphore,
				rendering_complete_semaphore,
				surface,
				depth_image_memory
			}
		}
	}
}

impl Drop for ExampleBase {
	fn drop(&mut self) {
		unsafe {
			self.device.device_wait_idle().unwrap();

			self.device.destroy_semaphore(self.present_complete_semaphore, None);
			self.device.destroy_semaphore(self.rendering_complete_semaphore, None);

			self.device.destroy_image_view(self.depth_image_view, None);
			self.device.destroy_image(self.depth_image, None);
			self.device.free_memory(self.depth_image_memory, None);

			for &image_view in self.present_image_views.iter() {
				self.device.destroy_image_view(image_view, None);
			}

			self.device.destroy_command_pool(self.pool, None);
			self.swapchain_loader.destroy_swapchain(self.swapchain, None);
		}
	}
}
