//! Quick and dirty code timing module.
//!
//! Measurements are `println`d out after `BUFFER_FLUSH_SIZE` measurements or after `flush` call.

use std::{
	fmt,
	ops::{Deref, DerefMut},
	time::{Duration, Instant}
};

pub const BUFFER_FLUSH_SIZE: usize = 1000;
pub const END_AFTER: usize = 10000;

type BufferType = Vec<(Duration, Duration)>;
struct BufferFmtWrapper(BufferType);
impl Deref for BufferFmtWrapper {
	type Target = BufferType;

	fn deref(&self) -> &Self::Target {
		&self.0
	}
}
impl DerefMut for BufferFmtWrapper {
	fn deref_mut(&mut self) -> &mut Self::Target {
		&mut self.0
	}
}
impl fmt::Debug for BufferFmtWrapper {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_list()
			.entries(
				self.0
					.iter()
					.map(|(all, middle)| (all.as_nanos(), middle.as_nanos()))
			)
			.finish()
	}
}

static mut BUFFER: BufferFmtWrapper = BufferFmtWrapper(Vec::new());
static mut LOOPS: usize = 0;

pub unsafe fn init() {
	BUFFER.reserve(BUFFER_FLUSH_SIZE);
}
pub unsafe fn flush() {
	if BUFFER.len() > 0 {
		println!("BUFFER: {:?}", BUFFER);
		BUFFER.set_len(0);
	}
}

#[inline(always)]
pub fn before() -> Instant {
	Instant::now()
}

#[inline(always)]
pub fn middle(before: Instant) -> [Instant; 2] {
	[before, Instant::now()]
}

#[inline(always)]
pub unsafe fn after(before_middle: [Instant; 2]) -> bool {
	let now = Instant::now();

	if LOOPS > END_AFTER {
		return false
	}

	let after_before = now.duration_since(before_middle[0]);
	let after_middle = now.duration_since(before_middle[1]);
	BUFFER.push((after_before, after_middle));

	let res = if BUFFER.len() == BUFFER_FLUSH_SIZE {
		flush();
		true
	} else {
		false
	};

	LOOPS += 1;
	if LOOPS > END_AFTER {
		if !res {
			flush();
			println!("@END@");
		}
		// panic!("End of the line");
	}

	res
}

macro_rules! sub_release {
	(
		$debug: expr,
		$release: expr
	) => {{
		#[cfg(debug_assertions)]
		let res = $debug;

		#[cfg(not(debug_assertions))]
		let res = $release;

		res
		}};
}
