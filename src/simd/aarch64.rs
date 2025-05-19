use std::arch::aarch64::*;
use std::mem::transmute;

pub struct Bytes {
    bitset: uint8x16x2_t,
}

type Vector = uint8x16_t;

/// Mapping from a number `i` in 0..=7 to a bit mask with the `i`-th bit set.
const N_TO_N_BITS_TABLE: uint8x16_t = unsafe {
    let mut bits = [0u8; 16];
    let mut i = 0u8;
    while i < 8 {
        bits[i as usize] = 1 << i;
        i += 1;
    }
    transmute(bits)
};

impl Bytes {
    pub fn new(bytes: [u8; 16], needle_len: i32) -> Self {
        assert!((0..=16).contains(&needle_len));
        let needle_len = needle_len as u8;

        // Make a bitset from the bytes to search for
        let mut bitset = [0u8; 256 / 8];
        for i in 0..needle_len {
            let i = usize::from(i);
            let value = usize::from(bytes[i]);
            let byte = value / 8;
            let bit = value % 8;
            let mask = 1 << bit;
            bitset[byte] |= mask;
        }
        let bitset = unsafe { transmute(bitset) };

        Bytes { bitset }
    }

    #[target_feature(enable = "neon")]
    pub fn find(&self, haystack: &[u8]) -> Option<usize> {
        let mut vectors = haystack.chunks_exact(size_of::<Vector>());

        let mut offset = 0;
        for vector in vectors.by_ref() {
            let vector = unsafe { vld1q_u8(vector.as_ptr()) };
            let result = self.locale_in_vector(vector);
            if let Some(element) = first_element_set(result) {
                return Some(offset + usize::from(element));
            }
            offset += size_of_val(&vector);
        }
        let remaining = vectors.remainder();
        let mut fake_vector = [0; size_of::<Vector>()];
        fake_vector[..remaining.len()].copy_from_slice(remaining);
        let vector = unsafe { vld1q_u8(fake_vector.as_ptr()) };
        let result = self.locale_in_vector(vector);
        if let Some(element) = first_element_set(result) {
            if element < remaining.len() as u8 {
                return Some(offset + usize::from(element));
            }
        }

        None
    }

    /// Given a vector of 16 bytes of input, return a vector of true/false values.
    ///
    /// Each element in the output will be 0/255 based on if the input byte at that position
    /// is equal to one of the values being searched for.
    #[inline]
    #[target_feature(enable = "neon")]
    fn locale_in_vector(&self, v: Vector) -> Vector {
        let high_bits = vshrq_n_u8::<3>(v);
        let low_bit_masks = vqtbl2q_u8(self.bitset, high_bits);
        let low_bits = vandq_u8(v, vdupq_n_u8(0b0111));
        let low_bits = vqtbl1q_u8(N_TO_N_BITS_TABLE, low_bits);
        vtstq_u8(low_bits, low_bit_masks)
    }
}

/// Convert a vector into a u64
///
/// Returns a value where the first 4 bits are the low 4 bits of the first element,
/// the next 4 bits are the high 4 bits of the second element, and so on.
///
/// For a "bool" vector (where every element is either 255 or 0), the resulting u64
/// will have groups of 4 bits from each element. e.g. the number of trailing zero bits
/// is 4 times the number of trailing zero elements.
#[target_feature(enable = "neon")]
fn vector_to_bitset(vector: Vector) -> u64 {
    let result_vector = vshrn_n_u16::<4>(vreinterpretq_u16_u8(vector));
    vget_lane_u64::<0>(vreinterpret_u64_u8(result_vector))
}

/// Find the first element in a "bool" vector that is set or None if all elements are zero
#[target_feature(enable = "neon")]
fn first_element_set(vector: Vector) -> Option<u8> {
    let bitset = vector_to_bitset(vector);
    let first_bit = bitset.trailing_zeros() / 4;
    if first_bit < 16 {
        Some(first_bit as u8)
    } else {
        None
    }
}
