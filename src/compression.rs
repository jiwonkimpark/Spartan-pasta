use std::fmt::{Debug, Formatter};
use ff::{Field, PrimeField};
use pasta_curves::{Ep, EpAffine, Fp};
use pasta_curves::arithmetic::CurveAffine;
use pasta_curves::group::Curve;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde::de::{SeqAccess, Visitor};

/// This file implements the compression and decompression of a point in elliptic curve
/// following the standard specified in https://www.secg.org/sec1-v2.pdf
/// Notice that this implementation is not generic and focused on the points in
/// Pasta Curves

/// The representation of a compressed point is an octet string M of length mlen
/// where mlen = \lceil (log q) / 8 \rceil + 1 and q is the order of the group.
/// Therefore, we construct a byte array with mlen = 32 + 1 = 33.

// TODO: make the following implementation as macro_rules! for both pallas and vesta curves

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct CompressedPoint(pub [u8; 33]);

impl CompressedPoint {
    /// If a point p = (x, y) is not an infinity point, point compression proceeds with the following steps:
    /// 1. convert x to an octet string m_x of length \lceil (log q) / 8 \rceil octets.
    /// 2. derive from y, a single bit \tilde{y} = y mod 2 and assign m_y = 0x02 or 0x03 based on \tilde{y}.
    /// 3. outputs m_y || m_x
    pub fn compress(p: Ep) -> CompressedPoint {
        let p_coordinates = p.to_affine().coordinates().unwrap();
        let m_y: u8 = Self::compress_y(p_coordinates.y());
        let m_x: [u8; 32] = p_coordinates.x().to_repr();

        let mut res = [0u8; 33];
        res[0] = m_y;
        res[1..33].copy_from_slice(&m_x);

        CompressedPoint(res)
    }

    /// To compress y, we first get \tilde{y_p} = y_p mod 2
    /// Then assign 0x02 to a single octet Y if \tilde{y_p} = 0
    /// And assign 0x03 to a single octet Y if \tilde{y_p} = 1
    fn compress_y(y: &Fp) -> u8 {
        let tilde_y = y.is_odd().unwrap_u8();

        match tilde_y {
            0 => { 2 }
            1 => { 3 }
            _ => { panic!("tilde_y should be either 0 or 1" )}
        }
    }

    /// Point decompression proceeds with the following steps:
    /// 1. parse m_y || m_x from the byte array
    /// 2. convert m_x to a field element x
    /// 3. convert m_y into \tilde{y}
    /// 4. derive from x and \tilde{y} an elliptic curve point P = (x, y)
    pub fn decompress(&self) -> Option<Ep> {
        let m_x = &self.0[1..33];
        let x = Fp::from_repr(<[u8; 32]>::try_from(m_x).unwrap()).unwrap();

        let m_y = self.0[0];
        let y = Self::decompress_y(m_y, x);

        let ep_affine = EpAffine::from_xy(x, y).unwrap();
        Some(Ep::from(ep_affine))
    }

    fn decompress_y(m_y: u8, x: Fp) -> Fp {
        let tilde_y = match m_y {
            2 => { 0u8 }
            3 => { 1u8 }
            _ => panic!("m_y should be either 2 or 3")
        };

        Self::y_from(x, tilde_y)
    }

    /// Get a corresponding y coordinate from x and \tilde{y}.
    /// First computes beta = \sqrt{alpha} = \sqrt{x^3 + ax + b} (mod p).
    /// Here we omit ax since a = 0 (mod p) for the Pasta curve.
    /// Then y = beta if beta == \tilde{y} mod 2
    /// and y = p - beta if beta != \tilde{y} mod 2
    fn y_from(x: Fp, tilde_y: u8) -> Fp {
        let x3 = x.square() * x;
        let five = Fp::from(5u64);
        let alpha = x3.add(&five);
        let beta = alpha.sqrt().unwrap();

        if beta.is_odd().unwrap_u8() == tilde_y {
            beta
        } else {
            beta.neg()
        }
    }

    pub fn to_bytes(&self) -> [u8; 32] {
        let mut bytes: [u8; 32] = [0u8; 32];

        bytes.copy_from_slice(&self.0[1..33]);

        bytes
    }

    pub const fn as_bytes(&self) -> &[u8; 33] {
        &self.0
    }
}

// Debug traits
impl Debug for CompressedPoint {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "CompressedPoint: {:?}", self.0)
    }
}

// implement Serialize and Deserialize
impl Serialize for CompressedPoint {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: Serializer {
        use serde::ser::SerializeTuple;

        let mut tup = serializer.serialize_tuple(33)?;
        for byte in self.as_bytes().iter() {
            tup.serialize_element(byte)?;
        }
        tup.end()
    }
}

impl<'de> Deserialize<'de> for CompressedPoint {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: Deserializer<'de> {
        struct PointVisitor;

        impl<'de> Visitor<'de> for PointVisitor {
            type Value = CompressedPoint;

            fn expecting(&self, formatter: &mut Formatter) -> std::fmt::Result {
                formatter.write_str("33 bytes of data")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error> where A: SeqAccess<'de> {
                let mut bytes = [0u8; 33];

                for i in 0..33 {
                    bytes[i] = seq.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(i, &"expected 33 bytes"))?;
                }

                Ok(CompressedPoint(bytes))

            }
        }

        deserializer.deserialize_tuple(33, PointVisitor)
    }
}

pub const PALLAS_GENERATOR_COMPRESSED: CompressedPoint = {
    let point: [u8; 33] = [2, 0, 0, 0, 0, 237, 48, 45, 153, 27, 249, 76, 9, 252, 152, 70, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64];
    CompressedPoint(point)
};

#[cfg(test)]
mod tests {
    use group::Group;
    use pasta_curves::Ep;
    use crate::compression::PALLAS_GENERATOR_COMPRESSED;
    use crate::group::GroupElement;

    #[test]
    fn compress_generator() {
        let generator = Ep::generator();
        let generator_group_element = GroupElement(generator);
        let compressed = generator_group_element.compress();

        println!("{:?}", compressed.0);
    }

    #[test]
    fn test_decompress() {
        let decompressed = PALLAS_GENERATOR_COMPRESSED.decompress().unwrap();
        let generator = Ep::generator();
        let generator_group_element = GroupElement(generator);

        assert_eq!(generator, decompressed);
        assert_eq!(generator_group_element, GroupElement(decompressed));
    }

    #[test]
    fn test_to_bytes() {
        let compressed = PALLAS_GENERATOR_COMPRESSED;
        let x_coordinates = compressed.to_bytes();

        assert_eq!(x_coordinates, compressed.0[1..33])
    }

    #[test]
    fn test_as_bytes() {
        let compressed = PALLAS_GENERATOR_COMPRESSED;
        let result = compressed.as_bytes();

        assert_eq!(compressed.0, *result)
    }
}