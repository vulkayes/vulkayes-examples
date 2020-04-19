use std::num::NonZeroU32;

use vulkayes_core::{
	ash::{self, vk},
	entry::Entry,
	instance::ApplicationInfo,
	prelude::*,
	swapchain::{image::SwapchainCreateImageInfo, SwapchainCreateInfo},
	util::fmt::VkVersion
};

use super::{frame::CommandState, BaseState, SwapchainState};

impl super::ApplicationState {
	/// Setups basics such as the instance, device and present queue.
	pub(super) fn setup_base<'a>(
		surface_extensions: impl Iterator<Item = &'a str>,
		surface_fn: impl FnOnce(Vrc<Instance>) -> Surface
	) -> (BaseState, Surface) {
		// Create Entry, which is the entry point into the Vulkan API.
		// The default entry is loaded as a dynamic library.
		let entry = Entry::new().unwrap();

		// Create instance from loaded entry
		// Also enable validation layers and require surface extensions as defined in vulkayes_window
		let layers = sub_release! {
			["VK_LAYER_KHRONOS_validation"].iter().map(|&s| s),
			std::iter::empty()
		};

		let instance = Instance::new(
			entry,
			ApplicationInfo {
				application_name: "VulkayesExample",
				engine_name: "VulkayesExample",
				api_version: VkVersion::new(1, 2, 0),
				..Default::default()
			},
			layers,
			surface_extensions.chain(std::iter::once(
				ash::extensions::ext::DebugReport::name().to_str().unwrap()
			)),
			Default::default(),
			// Lastly, register Default debug callbacks that log using the log crate
			vulkayes_core::instance::debug::DebugCallback::Default()
		)
		.expect("Could not create instance");

		// Create a surface
		let surface = surface_fn(instance.clone());
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
				.next()
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

		let base = BaseState {
			instance,
			device,
			present_queue
		};

		(base, surface)
	}

	/// Setups the initial create info for swapchain.
	pub(super) fn setup_swapchain_create_info(
		surface: &Surface,
		queue: &Queue,
		surface_size: [NonZeroU32; 2]
	) -> SwapchainCreateInfo<[u32; 1]> {
		let surface_format = surface
			.physical_device_surface_formats(queue.device().physical_device())
			.unwrap()
			.drain(..)
			.next()
			.expect("Unable to find suitable surface format.");
		let surface_capabilities = surface
			.physical_device_surface_capabilities(queue.device().physical_device())
			.unwrap();

		let desired_image_count = match surface_capabilities.max_image_count {
			0 => surface_capabilities.min_image_count + 1,
			a => (surface_capabilities.min_image_count + 1).min(a)
		};
		let surface_resolution = match surface_capabilities.current_extent.width {
			std::u32::MAX => surface_size,
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

	/// Setups the initial swapchain and present images
	pub(super) fn setup_swapchain(
		surface: Surface,
		present_queue: &Queue,
		size: [NonZeroU32; 2]
	) -> SwapchainState {
		let create_info = Self::setup_swapchain_create_info(&surface, &present_queue, size);

		let swapchain_data = Swapchain::new(
			present_queue.device().clone(),
			surface,
			create_info,
			Default::default()
		)
		.expect("could not create swapchain");

		let present_views = Self::create_present_views(swapchain_data.images);
		let framebuffers = Vec::with_capacity(present_views.len());

		let scissors = vk::Rect2D {
			offset: vk::Offset2D { x: 0, y: 0 },
			extent: vk::Extent2D {
				width: size[0].get(),
				height: size[1].get()
			}
		};
		let viewport = vk::Viewport {
			x: 0.0,
			y: 0.0,
			width: size[0].get() as f32,
			height: size[1].get() as f32,
			min_depth: 0.0,
			max_depth: 1.0
		};

		SwapchainState {
			create_info,
			swapchain: swapchain_data.swapchain,
			present_views,

			render_pass: None,
			framebuffers,

			scissors,
			viewport,

			outdated: 0,
			was_recreated: false
		}
	}

	pub(super) fn setup_commands(present_queue: &Queue) -> CommandState {
		let command_pool = CommandPool::new(
			present_queue,
			vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
			Default::default()
		)
		.expect("Could not create command pool");

		let mut command_buffers = CommandBuffer::new_multiple(
			command_pool.clone(),
			true,
			std::num::NonZeroU32::new(4).unwrap()
		)
		.expect("Could not allocate command buffers");

		let setup_command_buffer = command_buffers.pop().unwrap();
		let setup_fence = Fence::new(present_queue.device().clone(), false, Default::default())
			.expect("could not create fence");

		CommandState::new(
			command_pool,
			setup_command_buffer,
			setup_fence,
			command_buffers
		)
	}
}
