use vulkayes_core::{ash::vk, prelude::*, util::WaitTimeout};

#[derive(Debug)]
pub(super) struct FrameChain {
	command_buffer: Vrc<CommandBuffer>,

	// After image is acquired
	acquire_semaphore: BinarySemaphore,
	// After execution completes
	execution_semaphore: BinarySemaphore,
	execution_fence: Vrc<Fence>,

	pending: bool
}
impl FrameChain {
	pub fn new(command_pool: Vrc<CommandPool>) -> Self {
		let command_buffer = CommandBuffer::new(command_pool, true).expect("could not allocate command buffer");
		Self::from_command_buffer(command_buffer)
	}

	pub fn from_command_buffer(command_buffer: Vrc<CommandBuffer>) -> Self {
		let device = command_buffer.pool().device().clone();

		Self {
			command_buffer,
			acquire_semaphore: Semaphore::binary(device.clone(), Default::default()).expect("could not create semaphore"),
			execution_semaphore: Semaphore::binary(device.clone(), Default::default()).expect("could not create semaphore"),
			execution_fence: Fence::new(device, false, Default::default()).expect("could not create fence"),
			pending: false
		}
	}

	/// Returns `true` when `self` is ready to be used again.
	pub fn check_status(&mut self) -> bool {
		if !self.pending {
			true
		} else {
			if self.execution_fence.status().unwrap_or(false) {
				self.execution_fence.reset().expect("could not reset fence");
				self.pending = false;

				true
			} else {
				false
			}
		}
	}

	pub fn reset_command_buffer(&mut self) {
		unsafe {
			let cb_lock = self.command_buffer.lock().expect("vutex poisoned");
			self.command_buffer
				.pool()
				.device()
				.reset_command_buffer(
					*cb_lock,
					vk::CommandBufferResetFlags::RELEASE_RESOURCES
				)
				.expect("reset command buffer failed.");

			let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
			self.command_buffer
				.pool()
				.device()
				.begin_command_buffer(*cb_lock, &command_buffer_begin_info)
				.expect("could not begin command buffer");
		}
	}
}

#[derive(Debug)]
pub struct CommandState {
	command_pool: Vrc<CommandPool>,

	setup_command_buffer: Vrc<CommandBuffer>,
	setup_fence: Vrc<Fence>,

	draw_frame_chains: Vec<FrameChain>,
	next_chain: usize
}
impl CommandState {
	pub(super) fn new(
		command_pool: Vrc<CommandPool>,

		setup_command_buffer: Vrc<CommandBuffer>,
		setup_fence: Vrc<Fence>,

		draw_command_buffers: Vec<Vrc<CommandBuffer>>
	) -> Self {
		CommandState {
			command_pool,

			setup_command_buffer,
			setup_fence,

			draw_frame_chains: draw_command_buffers
				.into_iter()
				.map(|cb| FrameChain::from_command_buffer(cb))
				.collect(),
			next_chain: 0
		}
	}

	fn next_chain(&mut self) -> Option<usize> {
		let next = self.next_chain;
		if self.draw_frame_chains[next].check_status() {
			self.next_chain = (self.next_chain + 1) % self.draw_frame_chains.len();

			// self.draw_frame_chains[next].pending = true;
			return Some(next)
		}

		None
	}

	fn next_chain_or_new(&mut self) -> usize {
		if let Some(next) = self.next_chain() {
			return next
		}

		vulkayes_core::log::debug!("Creating new frame chain");
		let new_chain = FrameChain::new(self.command_pool.clone());
		if self.next_chain != 0 {
			// Reorder the new chains so that the next one is at the beginning
			let mut new_chains_vec = Vec::with_capacity(self.draw_frame_chains.len() + 1);
			new_chains_vec.extend(self.draw_frame_chains.drain(self.next_chain ..));
			new_chains_vec.extend(self.draw_frame_chains.drain(..));
			self.draw_frame_chains = new_chains_vec;
		}
		self.draw_frame_chains.push(new_chain);

		let next = self.draw_frame_chains.len() - 1;
		// self.draw_frame_chains[next].pending = true;
		next
	}

	pub(super) fn execute_setup_blocking(&mut self, queue: &Queue, setup_fn: impl FnOnce(&CommandBuffer)) {
		{
			let cb_lock = self.setup_command_buffer.lock().expect("vutex poisoned");
			unsafe {
				self.setup_command_buffer
					.pool()
					.device()
					.reset_command_buffer(
						*cb_lock,
						vk::CommandBufferResetFlags::RELEASE_RESOURCES
					)
					.expect("reset command buffer failed.");

				let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
				self.setup_command_buffer
					.pool()
					.device()
					.begin_command_buffer(*cb_lock, &command_buffer_begin_info)
					.expect("could not begin command buffer");
			}
		}

		setup_fn(&self.setup_command_buffer);

		{
			let cb_lock = self.setup_command_buffer.lock().expect("vutex poisoned");
			unsafe {
				self.setup_command_buffer
					.pool()
					.device()
					.end_command_buffer(*cb_lock)
					.expect("could not end command buffer");
			}
		}

		queue
			.submit(
				[],
				[],
				[&self.setup_command_buffer],
				[],
				Some(&self.setup_fence)
			)
			.expect("could not submit command buffer");

		self.setup_fence
			.wait(Default::default())
			.expect("could not wait for fence");
		self.setup_fence.reset().expect("could not reset fence");
	}
}

pub struct NextFrame<'a> {
	state: &'a mut super::ApplicationState,
	frame_index: usize,
	present_index: u32
}
impl NextFrame<'_> {
	fn chain(&self) -> &FrameChain {
		&self.state.command.draw_frame_chains[self.frame_index]
	}

	pub fn state(&self) -> &super::ApplicationState {
		self.state
	}

	pub fn present_image(&self) -> &Vrc<ImageView> {
		&self.state.swapchain.present_views[self.present_index as usize]
	}

	pub fn framebuffer(&self) -> vk::Framebuffer {
		self.state.swapchain.framebuffers[self.present_index as usize]
	}

	pub fn present_index(&self) -> u32 {
		self.present_index
	}

	pub fn command_buffer(&self) -> &Vrc<CommandBuffer> {
		&self.state.command.draw_frame_chains[self.frame_index].command_buffer
	}

	pub fn submit_and_present(mut self) {
		// vulkayes_core::const_queue_present!(
		// 	pub fn queue_present(
		// 		&queue,
		// 		waits: [&Semaphore; 1],
		// 		images: [&SwapchainImage; 1],
		// 		result_for_all: bool
		// 	) -> QueuePresentMultipleResult<[QueuePresentResult; _]>;
		// );

		// vulkayes_core::const_queue_submit! {
		// 	pub fn submit(
		// 		&queue,
		// 		waits: [&Semaphore; 1],
		// 		stages: [vk::PipelineStageFlags; _],
		// 		buffers: [&CommandBuffer; 1],
		// 		signals: [&Semaphore; 1],
		// 		fence: Option<&Fence>
		// 	) -> Result<(), QueueSubmitError>;
		// };

		unsafe {
			let cb_lock = self.command_buffer().lock().expect("vutex poisoned");
			self.command_buffer()
				.pool()
				.device()
				.end_command_buffer(*cb_lock)
				.expect("could not end command buffer");
		}

		if let Some(last_before) = self.state.last_before.take() {
			self.state.last_before_middle = Some(crate::dirty_mark::middle(last_before));
		}
		self.state
			.base
			.present_queue
			.submit(
				[&self.chain().acquire_semaphore],
				[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
				[self.command_buffer()],
				[&self.chain().execution_semaphore],
				Some(&self.chain().execution_fence)
			)
			.expect("queue submit failed");

		match self.state.base.present_queue.present(
			[&self.chain().execution_semaphore],
			[&self.state.swapchain.present_views[self.present_index as usize]
				.image()
				.try_as_swapchain_image()
				.unwrap()]
		) {
			Ok(vulkayes_core::queue::error::QueuePresentSuccess::SUCCESS) => (),
			Ok(vulkayes_core::queue::error::QueuePresentSuccess::SUBOPTIMAL_KHR)
			| Err(vulkayes_core::queue::error::QueuePresentError::ERROR_OUT_OF_DATE_KHR) => {
				self.state.swapchain.outdated += 1;
			}
			Err(e) => panic!("{}", e)
		}

		if crate::state::WAIT_AFTER_FRAME {
			self.state
				.base
				.present_queue
				.wait()
				.expect("could not wait for queue");
		}
	}
}

impl super::ApplicationState {
	pub fn execute_setup_blocking(&mut self, setup_fn: impl FnOnce(&CommandBuffer)) {
		self.command
			.execute_setup_blocking(&self.base.present_queue, setup_fn)
	}

	/// Acquires next swapchain image and returns next frame handle.
	pub fn acquire_next_frame(&mut self) -> NextFrame {
		self.check_and_recreate_swapchain();

		let frame_index = self.command.next_chain_or_new();

		let present_index = match self.swapchain.swapchain.acquire_next(
			WaitTimeout::Forever,
			(&self.command.draw_frame_chains[frame_index].acquire_semaphore).into()
		) {
			Ok(vulkayes_core::swapchain::error::AcquireResultValue::SUCCESS(i)) => i,
			Ok(vulkayes_core::swapchain::error::AcquireResultValue::SUBOPTIMAL_KHR(i)) => i,
			Err(vulkayes_core::swapchain::error::AcquireError::ERROR_OUT_OF_DATE_KHR) => {
				self.swapchain.outdated += 1;

				// recurse, eventually it will either be resolved or will panic
				return self.acquire_next_frame()
			}
			Err(e) => panic!("{}", e)
		};

		self.command.draw_frame_chains[frame_index].pending = true;
		self.command.draw_frame_chains[frame_index].reset_command_buffer();

		NextFrame { state: self, frame_index, present_index }
	}
}
