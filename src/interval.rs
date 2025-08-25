use std::{
    iter::Sum,
    ops::{Add, Mul},
};

type Num = f32;

#[derive(Clone, Copy, Debug)]
pub struct Interval(pub Num, pub Num);

// pub fn eq_tol(a: f32, b: f32) -> bool {
//     (a - b).abs() < 1e-9
// }

impl Interval {
    pub fn is_empty(&self) -> bool {
        self.0 > self.1
    }

    pub fn div_elems(self, x: Num) -> Interval {
        Interval(self.0 / x, self.1 / x)
    }

    pub fn mul_elems(self, x: Num) -> Interval {
        Interval(self.0 * x, self.1 * x)
    }

    #[allow(clippy::float_cmp)]
    pub fn is_singleton(&self) -> bool {
        self.0 == self.1
    }

    pub fn singleton(x: Num) -> Interval {
        Interval(x, x)
    }

    pub fn contains(&self, x: Num) -> bool {
        self.0 <= x && x <= self.1
    }

    pub fn intersects(&self, other: &Interval) -> bool {
        !(other.1 < self.0 || self.1 < other.0)
    }

    pub fn tighten(self, is_integer: bool) -> Interval {
        if !is_integer {
            return self;
        }
        Interval(self.0.ceil(), self.1.floor())
    }

    pub fn intersect(self, other: Interval) -> Interval {
        let lb = self.0.max(other.0);
        let ub = self.1.min(other.1);
        Interval(lb, ub)
    }

    pub fn eq(&self, other: &Interval) -> bool {
        self.0 == other.0 && self.1 == other.1
    }

    pub fn eq_tol(&self, other: &Interval) -> bool {
        (self.0 - other.0).abs() < 1e-6 && (self.1 - other.1).abs() < 1e-6
    }

    pub fn clamp(&self, x: Num) -> Num {
        // x.max(self.0).min(self.1)
        x.clamp(self.0, self.1)
    }
    pub fn is_subset_of(&self, other: &Interval) -> bool {
        other.0 <= self.0 && self.1 <= other.1
    }

    pub fn size(&self) -> Num {
        self.1 - self.0
    }
}

impl Mul<Interval> for Num {
    type Output = Interval;

    fn mul(self, rhs: Interval) -> Interval {
        if self >= 0. {
            Interval(self * rhs.0, self * rhs.1)
        } else {
            Interval(self * rhs.1, self * rhs.0)
        }
    }
}

impl Add for Interval {
    type Output = Interval;

    fn add(self, rhs: Self) -> Self::Output {
        Interval(self.0 + rhs.0, self.1 + rhs.1)
    }
}

impl Sum for Interval {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut i = Interval(0., 0.);
        for x in iter {
            i = i + x;
        }
        i
    }
}
