import ProjectEulerSolutions.Lib.SubstringDivisibility
import ProjectEulerSolutions.Lib.DigitMask

/-! The table lookup and mask states used by the Python substring-divisibility solver. -/
namespace ProjectEulerSolutions.SubstringSearch
open SubstringDivisibility

set_option Elab.async false

def multiples (p : Nat) : List Nat := (List.range (999/p+1)).map (p*·)
def hundreds (x : Nat) : Nat := x/100
def tens (x : Nat) : Nat := (x/10)%10
def units (x : Nat) : Nat := x%10
def distinct (x : Nat) : Bool :=
  decide (hundreds x ≠ tens x ∧ hundreds x ≠ units x ∧ tens x ≠ units x)
def digits (x : Nat) : List Nat := [hundreds x, tens x, units x]

/-- Values are appended in increasing order of the original three-digit multiple. -/
def table (p b c : Nat) : List Nat :=
  ((multiples p).filter fun x => distinct x && decide (tens x = b ∧ units x = c)).map hundreds

abbrev State := List Nat × Nat
def coherent (state : State) : Prop := state.2 = DigitMask.mask state.1

def seeds : List State := ((multiples 17).filter distinct).map fun x =>
  (digits x, (2^hundreds x) ||| (2^tens x) ||| (2^units x))

def extend (p : Nat) (states : List State) : List State :=
  states.flatMap fun (suffix, used) =>
    ((table p (suffix.headD 0) ((suffix.drop 1).headD 0)).filter
      fun digit => decide (used &&& 2^digit = 0)).map fun digit =>
        (digit :: suffix, used ||| (2^digit))

def states : List State := [13, 11, 7, 5, 3, 2].foldl (fun previous p => extend p previous) seeds

def complete (state : State) : List (List Nat) :=
  let remaining := 1023 ^^^ state.2
  if remaining = 0 ∨ remaining &&& (remaining-1) ≠ 0 then []
  else
    let first := remaining.size-1
    if first = 0 then [] else [first :: state.1]

def rows : List (List Nat) := states.flatMap complete

theorem seeds_coherent (state : State) (member : state ∈ seeds) : coherent state := by
  obtain ⟨x, _, rfl⟩ := List.mem_map.mp member
  simp [coherent, DigitMask.mask, digits, Nat.or_assoc]

theorem extend_coherent (p : Nat) (previous : List State)
    (valid : ∀ state ∈ previous, coherent state) (state : State)
    (member : state ∈ extend p previous) : coherent state := by
  obtain ⟨prior, inside, next⟩ := List.mem_flatMap.mp member
  obtain ⟨digit, _, rfl⟩ := List.mem_map.mp next
  change prior.2 ||| 2^digit = 2^digit ||| DigitMask.mask prior.1
  rw [show prior.2 = DigitMask.mask prior.1 from valid prior inside, Nat.or_comm]

theorem states_coherent (state : State) (member : state ∈ states) : coherent state := by
  have preserved (ps : List Nat) (previous : List State)
      (valid : ∀ state ∈ previous, coherent state) :
      ∀ state ∈ ps.foldl (fun previous p => extend p previous) previous, coherent state := by
    induction ps generalizing previous with
    | nil => exact valid
    | cons p ps ih => exact ih _ (extend_coherent p previous valid)
  exact preserved _ seeds seeds_coherent state member

set_option maxRecDepth 4000 in
set_option maxHeartbeats 2000000 in
/-- The small suffix search is a complete enumeration of the nine trailing digits. -/
theorem suffixes_perm : (states.map Prod.fst).Perm
    (ConstraintRows.search pool SubstringDivisibility.tests.tail) := by
  decide +kernel

set_option maxRecDepth 4000 in
set_option maxHeartbeats 2000000 in
/-- Completing the mask states agrees with the general constrained-permutation search. -/
theorem rows_perm_search : rows.Perm SubstringDivisibility.rows := by
  decide +kernel

/-- No exhaustive enumeration is evaluated by the kernel: completeness follows from the general theorem. -/
theorem rows_perm_reference : rows.Perm referenceRows :=
  rows_perm_search.trans SubstringDivisibility.rows_perm_reference

theorem sum_equiv : (rows.map value).sum = (referenceRows.map value).sum :=
  (rows_perm_reference.map value).sum_eq

end ProjectEulerSolutions.SubstringSearch
