import Mathlib.Data.Nat.Bitwise
import Mathlib.Data.Nat.Size
import Mathlib.Data.List.Perm.Basic

/-! The bit set of a digit sequence, independently of its order or multiplicities. -/
namespace ProjectEulerSolutions.DigitMask

set_option Elab.async false

def mask : List Nat → Nat
  | [] => 0
  | digit :: digits => (2^digit) ||| mask digits

theorem testBit_mask (digits : List Nat) (index : Nat) :
    (mask digits).testBit index = decide (index ∈ digits) := by
  induction digits with
  | nil => simp [mask]
  | cons digit digits ih => simp [mask, Nat.testBit_two_pow, ih, eq_comm]

theorem mask_perm {left right : List Nat} (permuted : left.Perm right) : mask left = mask right := by
  apply Nat.eq_of_testBit_eq
  intro index
  simp only [testBit_mask, permuted.mem_iff]

/-- Testing a single digit bit gives that bit on a hit and zero on a miss. -/
theorem and_bit (digits : List Nat) (digit : Nat) :
    mask digits &&& 2^digit = if digit ∈ digits then 2^digit else 0 := by
  apply Nat.eq_of_testBit_eq
  intro index
  by_cases same : digit = index
  · subst index
    by_cases member : digit ∈ digits <;> simp [testBit_mask, member]
  · by_cases member : digit ∈ digits <;> simp [testBit_mask, member, same]

theorem and_bit_ne_zero (digits : List Nat) (digit : Nat) :
    mask digits &&& 2^digit ≠ 0 ↔ digit ∈ digits := by
  rw [and_bit]
  by_cases member : digit ∈ digits <;> simp [member]

/-- XOR removes a fresh prepended digit, leaving precisely its missing bit. -/
theorem xor_tail (digit : Nat) (digits : List Nat) (fresh : digit ∉ digits) :
    mask (digit :: digits) ^^^ mask digits = 2^digit := by
  apply Nat.eq_of_testBit_eq
  intro index
  by_cases same : digit = index
  · subst index
    simp [mask, testBit_mask, fresh]
  · simp [mask, testBit_mask, same]

theorem missing_bit (pool : List Nat) (digit : Nat) (digits : List Nat)
    (distinct : pool.Nodup) (permuted : (digit :: digits).Perm pool) :
    mask pool ^^^ mask digits = 2^digit := by
  rw [← mask_perm permuted]
  exact xor_tail digit digits (List.nodup_cons.mp (permuted.nodup_iff.mpr distinct)).1

theorem missing_bit_size (pool : List Nat) (digit : Nat) (digits : List Nat)
    (distinct : pool.Nodup) (permuted : (digit :: digits).Perm pool) :
    (mask pool ^^^ mask digits).size - 1 = digit := by
  rw [missing_bit pool digit digits distinct permuted, Nat.size_pow]
  omega

end ProjectEulerSolutions.DigitMask
