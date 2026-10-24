import lang.py.builtin.type.bytearray
import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Tauto

/-! Shared mathematical invariant and exact slice semantics for bytearray sieves. -/
namespace ProjectEulerSolutions.PrimeSieve

/-- Candidates below `processed` have removed their composite multiples. -/
def Survives (processed value : Nat) : Prop :=
  ∀ p, p < processed → Nat.Prime p → p^2 ≤ value → ¬ p ∣ value

theorem prime_survives (processed value : Nat) (prime : Nat.Prime value) : Survives processed value := by
  intro p _ pp square divides
  rcases (Nat.dvd_prime prime).mp divides with one | same
  · have := pp.two_le; omega
  · have := pp.two_le; nlinarith

theorem survives_current (p : Nat) (lower : 2 ≤ p) : Survives p p ↔ Nat.Prime p := by
  refine ⟨?_, prime_survives p p⟩
  intro survives
  by_contra composite
  have prime := Nat.minFac_prime (by omega : p ≠ 1)
  have square := Nat.minFac_sq_le_self (by omega : 0 < p) composite
  have before : p.minFac < p := by have := prime.two_le; nlinarith
  exact survives p.minFac before prime square (Nat.minFac_dvd p)

theorem survives_finished (limit p value : Nat) (lower : 2 ≤ value) (upper : value ≤ limit)
    (finished : limit < p^2) : Survives p value ↔ Nat.Prime value := by
  refine ⟨?_, prime_survives _ _⟩
  intro survives
  by_contra composite
  have prime := Nat.minFac_prime (by omega : value ≠ 1)
  have square := Nat.minFac_sq_le_self (by omega : 0 < value) composite
  have before : value.minFac < p := by nlinarith
  exact survives value.minFac before prime square (Nat.minFac_dvd value)

theorem survives_next (p value : Nat) :
    Survives (p+1) value ↔ Survives p value ∧ (Nat.Prime p → p^2 ≤ value → ¬ p ∣ value) := by
  constructor
  · intro h; exact ⟨fun q before => h q (by omega), h p (by omega)⟩
  · rintro ⟨h, current⟩ q before
    by_cases same : q = p
    · simpa [same] using current
    · exact h q (by omega)

def markedIndices (length p : Nat) : List Nat := List.range' (p*p) ((length-p*p+p-1)/p) p
def mark (bytes : List UInt8) (p : Nat) : List UInt8 :=
  (markedIndices bytes.length p).foldl (fun result i => result.set i 0) bytes

theorem mem_markedIndices (length p i : Nat) (positive : 0 < p) :
    i ∈ markedIndices length p ↔ i < length ∧ p^2 ≤ i ∧ p ∣ i := by
  rw [markedIndices, FormalSW.Py.Builtin.Slice.mem_positive_end _ _ _ _ positive]
  have divisible (lower : p*p ≤ i) : p ∣ i-p*p ↔ p ∣ i :=
    Nat.dvd_sub_iff_left lower (dvd_mul_right p p)
  constructor
  · rintro ⟨lower, upper, divides⟩
    exact ⟨upper, by nlinarith, (divisible lower).mp divides⟩
  · rintro ⟨upper, square, divides⟩
    have lower : p*p ≤ i := by nlinarith
    exact ⟨lower, upper, (divisible lower).mpr divides⟩

theorem mark_length (bytes : List UInt8) (p : Nat) : (mark bytes p).length = bytes.length := by
  unfold mark
  generalize markedIndices bytes.length p = indices
  induction indices generalizing bytes with
  | nil => rfl
  | cons i rest ih => simpa only [List.foldl_cons, List.length_set] using ih (bytes.set i 0)

theorem mark_get (bytes : List UInt8) (p i : Nat) (positive : 0 < p) (valid : i < bytes.length) :
    (mark bytes p)[i]? = if p^2 ≤ i ∧ p ∣ i then some 0 else bytes[i]? := by
  rw [mark, FormalSW.Py.Builtin.List.getElem?_setUniform _ _ _ _ valid]
  simp only [mem_markedIndices _ _ _ positive, valid, true_and]

def Sieved (limit processed : Nat) (bytes : List UInt8) : Prop :=
  bytes.length = limit+1 ∧ ∀ i byte, 2 ≤ i → bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Survives processed i)

def initial (limit : Nat) : List UInt8 := [0,0] ++ List.replicate (limit-1) 1

theorem sieved_initial (limit : Nat) (large : 2 ≤ limit) : Sieved limit 2 (initial limit) := by
  refine ⟨by simp [initial]; omega, ?_⟩
  intro i byte lower found
  have equal : byte = 1 := by
    have tail : (List.replicate (limit-1) (1 : UInt8))[i-2]? = some byte := by
      rw [show i = (i-2)+2 by omega] at found
      simpa [initial] using found
    exact List.eq_of_mem_replicate (List.mem_of_getElem? tail)
  subst byte
  simp only [UInt8.toNat_ofNat, Nat.one_mod, ne_eq, one_ne_zero, not_false_eq_true, true_iff]
  intro p before prime
  have := prime.two_le
  omega

theorem sieved_skip (limit p : Nat) (bytes : List UInt8) (sieved : Sieved limit p bytes)
    (composite : ¬ Nat.Prime p) : Sieved limit (p+1) bytes := by
  refine ⟨sieved.1, ?_⟩
  intro i byte lower found
  rw [sieved.2 i byte lower found, survives_next]
  simp [composite]

theorem sieved_mark (limit p : Nat) (bytes : List UInt8) (sieved : Sieved limit p bytes)
    (prime : Nat.Prime p) : Sieved limit (p+1) (mark bytes p) := by
  refine ⟨by rw [mark_length, sieved.1], ?_⟩
  intro i byte lower found
  have valid : i < bytes.length := by
    have := (List.getElem?_eq_some_iff.mp found).1
    simpa only [mark_length] using this
  rw [mark_get bytes p i prime.pos valid] at found
  rw [survives_next]
  by_cases removed : p^2 ≤ i ∧ p ∣ i
  · simp only [if_pos removed, Option.some.injEq] at found
    subst byte
    simp [prime, removed.1, removed.2]
  · simp only [if_neg removed] at found
    rw [sieved.2 i byte lower found]
    simp only [prime, true_implies]
    tauto

theorem mark_slice (bytes : List UInt8) (p : Nat) (lower : 2 ≤ p) (inside : p*p < bytes.length) :
    FormalSW.Py.Builtin.ByteArray.setSlice bytes
      { start := some (p*p), stop := some bytes.length, step := some p }
      (List.replicate (((bytes.length : Int)-p*p-1)/p+1).toNat 0) = .ok (mark bytes p) := by
  have positive : 0 < p := by omega
  have normalized : FormalSW.Py.Builtin.Slice.indices
      { start := some (p*p), stop := some bytes.length, step := some p } bytes.length =
      .ok (((p*p : Nat) : Int), (bytes.length : Int), (p : Int)) := by
    have nonzero : (p : Int) ≠ 0 := by omega
    simp only [FormalSW.Py.Builtin.Slice.indices, Option.getD_some, nonzero, ↓reduceIte,
      show (0 : Int) < p by omega]
    change Except.ok
      (max 0 (min (if (p : Int)*p < 0 then (p : Int)*p + bytes.length else (p : Int)*p) bytes.length),
       max 0 (min (if (bytes.length : Int) < 0 then (bytes.length : Int)+bytes.length else bytes.length) bytes.length),
       (p : Int)) = _
    simp only [show ¬ (p : Int)*p < 0 by exact not_lt.mpr (mul_self_nonneg _), show ¬ (bytes.length : Int) < 0 by omega, if_false]
    simp only [Nat.cast_mul]
    have nonnegative : 0 ≤ (p : Int)*p := mul_self_nonneg _
    simp [min_eq_left (show (p : Int)*p ≤ bytes.length by exact_mod_cast Nat.le_of_lt inside), max_eq_right nonnegative]
  have selected := FormalSW.Py.Builtin.Slice.indexList_positive_end bytes.length (p*p) p positive
  simp only [Nat.cast_mul] at selected
  have selected' : FormalSW.Py.Builtin.Slice.indexList
      { start := some (p*p), stop := some bytes.length, step := some p } bytes.length =
      .ok (markedIndices bytes.length p) := by
    have normalizedNone : FormalSW.Py.Builtin.Slice.indices
        { start := some (p*p), step := some p } bytes.length =
        .ok (((p*p : Nat) : Int), (bytes.length : Int), (p : Int)) := by
      have nonzero : (p : Int) ≠ 0 := by omega
      simp only [FormalSW.Py.Builtin.Slice.indices, Option.getD_some, nonzero, ↓reduceIte,
        show (0 : Int) < p by omega]
      change Except.ok
        (max 0 (min (if (p : Int)*p < 0 then (p : Int)*p + bytes.length else (p : Int)*p) bytes.length),
         (bytes.length : Int), (p : Int)) = _
      simp only [show ¬ (p : Int)*p < 0 by exact not_lt.mpr (mul_self_nonneg _), if_false]
      simp only [Nat.cast_mul]
      congr 2; omega
    simpa only [FormalSW.Py.Builtin.Slice.indexList, normalizedNone, normalized, markedIndices] using selected
  have count : (((bytes.length : Int)-p*p-1)/p+1).toNat = (markedIndices bytes.length p).length := by
    rw [markedIndices, List.length_range']
    have casts : ((p*p : Nat) : Int) = (p : Int)*p := by simp
    rw [show (bytes.length : Int)-p*p-1 = (bytes.length : Int)-1-(p*p : Nat) by simp; ring]
    exact FormalSW.Py.Builtin.Slice.positive_end_count _ _ _ positive
  rw [count]
  exact FormalSW.Py.Builtin.ByteArray.setSlice_uniform bytes _ 0 normalized (by omega) selected'

end ProjectEulerSolutions.PrimeSieve
