import ProjectEulerSolutions.Lib.DigitMask
import Mathlib.Data.List.Sublists
import Mathlib.Algebra.BigOperators.Group.List.Basic
import Mathlib.Tactic

/-! Digit-labeled dice, rotation of 6 and 9, and unordered pairs of distinct designs. -/
namespace ProjectEulerSolutions.DigitDice
set_option Elab.async false

/-- Ordered selections of positions, keeping the original printed labels. -/
def choices : Nat → List α → List (List α)
  | 0, _ => [[]]
  | _ + 1, [] => []
  | count + 1, first :: rest =>
      (choices count rest).map (first :: ·) ++ choices (count + 1) rest
termination_by _ pool => pool.length

theorem choices_eq_reverse_sublistsLen (count : Nat) (pool : List α) :
    choices count pool = (pool.sublistsLen count).reverse := by
  induction pool generalizing count with
  | nil => cases count <;> simp [choices]
  | cons first rest ih =>
      cases count with
      | zero => simp [choices]
      | succ count => simp [choices, List.sublistsLen_succ_cons, List.reverse_append, List.map_reverse, ih]

theorem choices_length (count : Nat) (pool : List α) :
    (choices count pool).length = pool.length.choose count := by
  rw [choices_eq_reverse_sublistsLen, List.length_reverse, List.length_sublistsLen]

/-- More faces than available labels give no designs, independent of any enumeration limit. -/
theorem choices_too_large (count : Nat) (pool : List α) (large : pool.length < count) :
    choices count pool = [] := by
  apply List.length_eq_zero_iff.mp
  rw [choices_length, Nat.choose_eq_zero_of_lt large]

def addLabel (labels : List Nat) (label : Nat) : List Nat :=
  if label ∈ labels then labels else labels ++ [label]

theorem mem_addLabel (labels : List Nat) (label digit : Nat) :
    digit ∈ addLabel labels label ↔ digit ∈ labels ∨ digit = label := by
  by_cases present : label ∈ labels <;> simp [addLabel, present]
  intro equal
  simpa [equal] using present

def rotated (labels : List Nat) : List Nat :=
  if 6 ∈ labels ∨ 9 ∈ labels then addLabel (addLabel labels 6) 9 else labels

def Shows (labels : List Nat) (digit : Nat) : Prop :=
  digit ∈ labels ∨ ((digit = 6 ∨ digit = 9) ∧ (6 ∈ labels ∨ 9 ∈ labels))

theorem mem_rotated (labels : List Nat) (digit : Nat) : digit ∈ rotated labels ↔ Shows labels digit := by
  by_cases active : 6 ∈ labels ∨ 9 ∈ labels <;> simp [rotated, active, mem_addLabel, Shows, or_assoc]

def rotateMask (mask : Nat) : Nat :=
  if mask.testBit 6 || mask.testBit 9 then mask ||| (2 ^ 6 ||| 2 ^ 9) else mask

/-- Bit expansion and list expansion describe exactly the same displayable labels. -/
theorem rotateMask_mask (labels : List Nat) : rotateMask (DigitMask.mask labels) = DigitMask.mask (rotated labels) := by
  apply Nat.eq_of_testBit_eq
  intro digit
  have bits : (576 : Nat).testBit digit = (decide (digit = 6) || decide (digit = 9)) := by
    change (2 ^ 6 ||| 2 ^ 9).testBit digit = _
    rw [Nat.testBit_or]
    simp only [Nat.testBit_two_pow, eq_comm]
  by_cases six : 6 ∈ labels <;> by_cases nine : 9 ∈ labels <;>
    simp [rotateMask, DigitMask.testBit_mask, rotated, six, nine, mem_addLabel,
      bits, Bool.or_assoc]

theorem mask_bit_test (labels : List Nat) (digit : Nat) :
    (DigitMask.mask labels &&& 2 ^ digit != 0) = decide (digit ∈ labels) := by
  by_cases present : digit ∈ labels <;> simp [DigitMask.and_bit, present]

def displays (left right : List Nat) (required : List (Nat × Nat)) : Bool :=
  required.all fun (x, y) =>
    (decide (x ∈ rotated left) && decide (y ∈ rotated right)) ||
      (decide (y ∈ rotated left) && decide (x ∈ rotated right))

def maskDisplays (left right : Nat) (required : List (Nat × Nat)) : Bool :=
  required.all fun (x, y) =>
    (left &&& 2 ^ x != 0) && (right &&& 2 ^ y != 0) ||
      (left &&& 2 ^ y != 0) && (right &&& 2 ^ x != 0)

theorem maskDisplays_rotated (left right : List Nat) (required : List (Nat × Nat)) :
    maskDisplays (rotateMask (DigitMask.mask left)) (rotateMask (DigitMask.mask right)) required =
      displays left right required := by
  rw [rotateMask_mask, rotateMask_mask]
  simp only [maskDisplays, displays, mask_bit_test]

/-- Positions represent distinct printed designs even when their display masks coincide. -/
def countPairs (test : α → α → Bool) : List α → Nat
  | [] => 0
  | first :: rest => (first :: rest).countP (test first) + countPairs test rest

theorem countPairs_map (test : β → β → Bool) (f : α → β) (items : List α) :
    countPairs test (items.map f) = countPairs (fun left right => test (f left) (f right)) items := by
  induction items with
  | nil => rfl
  | cons first rest ih => simp [countPairs, List.countP_cons, List.countP_map, Function.comp_def, ih]

def referenceCount (faceCount : Nat) (required : List (Nat × Nat)) : Nat :=
  countPairs (fun left right => displays left right required) (choices faceCount (List.range 10))

def optimizedCount (faceCount : Nat) (required : List (Nat × Nat)) : Nat :=
  countPairs (fun left right => maskDisplays left right required)
    ((choices faceCount (List.range 10)).map (fun labels => rotateMask (DigitMask.mask labels)))

theorem optimizedCount_eq (faceCount : Nat) (required : List (Nat × Nat)) :
    optimizedCount faceCount required = referenceCount faceCount required := by
  simp only [optimizedCount, countPairs_map, maskDisplays_rotated, referenceCount]

theorem referenceCount_too_large (faceCount : Nat) (required : List (Nat × Nat)) (large : 10 < faceCount) :
    referenceCount faceCount required = 0 := by
  rw [referenceCount, choices_too_large faceCount (List.range 10) (by simpa using large)]
  rfl

/-- Signed face counts have no designs below zero; oversized natural selections are already empty. -/
def answer (faceCount : Int) (required : List (Nat × Nat)) : Nat :=
  if faceCount < 0 then 0 else referenceCount faceCount.toNat required

theorem answer_invalid (faceCount : Int) (required : List (Nat × Nat))
    (impossible : faceCount < 0 ∨ 10 < faceCount) : answer faceCount required = 0 := by
  by_cases negative : faceCount < 0
  · simp [answer, negative]
  · have large : 10 < faceCount.toNat := by omega
    simp [answer, negative, referenceCount_too_large _ _ large]

end ProjectEulerSolutions.DigitDice
