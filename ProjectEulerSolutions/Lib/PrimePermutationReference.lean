import ProjectEulerSolutions.Lib.PrimePermutationSearch

/-! The brute-force reference has exactly the two certified four-digit triples. -/
namespace ProjectEulerSolutions.PrimePermutationReference
open ProjectEulerStatements.P49 ProjectEulerSolutions.PrimePermutationGroups

set_option Elab.async false

def accepted (a d : Nat) : Bool :=
  decide (1000≤a ∧ 0<d ∧ a+2*d<10000) && allPrime (seqFrom a d 3) && allPerm (seqFrom a d 3)

def certifiedAccepted (a d : Nat) : Bool :=
  decide (1000≤a ∧ 0<d ∧ a+2*d<10000) &&
    [a,a+d,a+2*d].all SixStepPrimality.certifiedPrime &&
    [a+d,a+2*d].all (isPermutation a)

theorem certifiedAccepted_correct (a d : Nat) : certifiedAccepted a d=accepted a d := by
  simp [certifiedAccepted, accepted, allPrime, allPerm, seqFrom, isPrime,
    List.range_succ, SixStepPrimality.certifiedPrime_correct]

set_option maxRecDepth 8192 in
private theorem known_accepted : accepted 1487 3330=true := by
  rw [← certifiedAccepted_correct]
  simp only [certifiedAccepted, List.all_cons, List.all_nil, Bool.and_true]
  simp only [isPermutation, digitsSorted]
  norm_num [Nat.digits_of_two_le_of_pos]
  simp only [List.mergeSort_eq_insertionSort]
  decide +kernel

set_option maxRecDepth 8192 in
private theorem winner_accepted : accepted 2969 3330=true := by
  rw [← certifiedAccepted_correct]
  simp only [certifiedAccepted, List.all_cons, List.all_nil, Bool.and_true]
  simp only [isPermutation, digitsSorted]
  norm_num [Nat.digits_of_two_le_of_pos]
  simp only [List.mergeSort_eq_insertionSort]
  decide +kernel

/-- The exhaustive pair search has only two possible nonempty iterations. -/
theorem accepted_iff (a d : Nat) : accepted a d=true ↔
    (a=1487 ∧ d=3330) ∨ (a=2969 ∧ d=3330) := by
  constructor
  · intro accepts
    have properties : (1000≤a ∧ 0<d ∧ a+2*d<10000) ∧
        (Nat.Prime a ∧ Nat.Prime (a+d) ∧ Nat.Prime (a+2*d)) ∧
        (isPermutation a (a+d)=true ∧ isPermutation a (a+2*d)=true) := by
      simpa [accepted, allPrime, allPerm, seqFrom, isPrime, List.range_succ, and_assoc] using accepts
    have classified := PrimePermutationSearch.classifies a (a+d) (a+2*d)
      properties.1.1 properties.1.2.2 (by omega) (by omega) (by omega)
      properties.2.1.1 properties.2.1.2.1 properties.2.1.2.2 properties.2.2.1 properties.2.2.2
    rcases classified with row | row
    · left
      change (a,a+d,a+2*d)=(1487,4817,8147) at row
      have fields := Prod.mk.inj row
      have rest := Prod.mk.inj fields.2
      omega
    · right
      change (a,a+d,a+2*d)=(2969,6299,9629) at row
      have fields := Prod.mk.inj row
      have rest := Prod.mk.inj fields.2
      omega
  · rintro (⟨rfl,rfl⟩ | ⟨rfl,rfl⟩)
    · exact known_accepted
    · exact winner_accepted

private theorem collect_eq (indices : List Nat) (predicate : Nat → Bool) (row : Nat → List Nat)
    (tail : List (List Nat)) :
    indices.foldr (fun i acc => if predicate i then row i :: acc else acc) tail =
      ((indices.filter predicate).map row)++tail := by
  induction indices with
  | nil => rfl
  | cons i indices ih =>
      simp only [List.foldr_cons, List.filter_cons]
      split <;> simp_all

private theorem differences (a : Nat) (tail : List (List Nat)) :
    (List.range 10000).foldr (fun d acc => if accepted a d then seqFrom a d 3 :: acc else acc) tail =
      if a=1487 ∨ a=2969 then seqFrom a 3330 3 :: tail else tail := by
  rw [collect_eq]
  by_cases eligible : a=1487 ∨ a=2969
  · have selected : (List.range 10000).filter (accepted a)=[3330] := by
      apply (List.pairwise_lt_range.filter (accepted a)).eq_of_mem_iff (by simp)
      intro d
      simp only [List.mem_filter, List.mem_range, List.mem_singleton, accepted_iff]
      omega
    rw [selected, if_pos eligible]
    rfl
  · have empty : (List.range 10000).filter (accepted a)=[] := by
      apply List.filter_eq_nil_iff.mpr
      intro d _
      change ¬ accepted a d=true
      rw [accepted_iff]
      omega
    rw [empty, if_neg eligible]
    rfl

/-- The mathematical reference's original fold order lists the given example first. -/
theorem reference_rows : naive 4 3=[[1487,4817,8147],[2969,6299,9629]] := by
  change (List.range 10000).foldr (fun a acc =>
    (List.range 10000).foldr (fun d acc2 => if accepted a d then seqFrom a d 3 :: acc2 else acc2) acc) [] = _
  simp_rw [differences]
  simp_rw [show ∀ (a : Nat) (acc : List (List Nat)),
      (if a=1487 ∨ a=2969 then seqFrom a 3330 3 :: acc else acc) =
      (if decide (a=1487 ∨ a=2969) then seqFrom a 3330 3 :: acc else acc) from
        fun a acc => by simp only [decide_eq_true_eq]]
  rw [collect_eq]
  have selected : (List.range 10000).filter (fun a => decide (a=1487 ∨ a=2969))=[1487,2969] := by
    apply (List.pairwise_lt_range.filter _).eq_of_mem_iff (by simp)
    intro a
    simp only [List.mem_filter, List.mem_range, decide_eq_true_eq, List.mem_cons, List.not_mem_nil, or_false]
    omega
  rw [selected]
  rfl

end ProjectEulerSolutions.PrimePermutationReference
