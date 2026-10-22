#![no_std]
#![no_main]

use defmt_rtt as _;
use panic_probe as _;
use rp2040_hal as _;

#[global_allocator]
static HEAP: embedded_alloc::LlffHeap = embedded_alloc::LlffHeap::empty();

#[link_section = ".boot2"]
#[used]
pub static BOOT2: [u8; 256] = rp2040_boot2::BOOT_LOADER_GENERIC_03H;

#[cfg(test)]
#[embedded_test::tests]
mod tests {
    #[init]
    fn init() {
        let memory = cortex_m::singleton!(: [u8; 16 * 1024] = [0; 16 * 1024]).unwrap();
        unsafe { super::HEAP.init(memory.as_mut_ptr() as usize, memory.len()) };
    }

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
