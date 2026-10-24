import ProjectEulerSolutions.Lib.CliqueSearchCorrectness

namespace ProjectEulerSolutions.CliqueSearch

theorem member_le_sum (values : List Nat) (p : Nat) (member : p ∈ values) : p ≤ values.sum := by
  induction values with
  | nil => simp at member
  | cons head rest ih =>
    rcases List.mem_cons.mp member with same | inside
    · subst p
      simp
    · have lower := ih inside
      simp only [List.sum_cons]
      omega

/-- Only increasing pairs from the actual vertex list are relevant. -/
def AgreeBelow (left right : Nat → Nat → Bool) (values : List Nat) (bound : Nat) : Prop :=
  ∀ p ∈ values, ∀ q ∈ values, p < q → q ≤ bound → left p q = right p q

theorem AgreeBelow.symm {left right : Nat → Nat → Bool} {values : List Nat} {bound : Nat}
    (agreement : AgreeBelow left right values bound) : AgreeBelow right left values bound :=
  fun p memberP q memberQ order bounded => (agreement p memberP q memberQ order bounded).symm

theorem Completion.change_relation {left right : Nat → Nat → Bool} {values chosen : List Nat} {need bound : Nat}
    (completion : Completion left need values chosen) (ordered : values.Pairwise (· < ·))
    (agreement : AgreeBelow left right values bound) (bounded : chosen.sum ≤ bound) :
    Completion right need values chosen := by
  refine ⟨completion.size, completion.sublist, ?_⟩
  have sorted := List.Pairwise.sublist completion.sublist ordered
  apply (completion.compatible.and sorted).imp_of_mem
  intro p q memberP memberQ pair
  rw [← agreement p (completion.sublist.mem memberP) q (completion.sublist.mem memberQ) pair.2
    (Nat.le_trans (member_le_sum chosen q memberQ) bounded)]
  exact pair.1

/-- Once an improving witness bounds the result, relation agreement below
that sum suffices. More expensive completions cannot change the minimum or
its tie selection, irrespective of the size of the original vertex list. -/
theorem Optimal.change_relation_below {left right : Nat → Nat → Bool} {need : Nat} {values : List Nat}
    {before after : Best} {bound : Nat} (optimal : Optimal left need [] values 0 before after)
    (ordered : values.Pairwise (· < ·)) (agreement : AgreeBelow left right values bound)
    (bounded : after.bound ≤ (bound : Int)) : Optimal right need [] values 0 before after := by
  constructor
  · rcases optimal.sound with unchanged | ⟨chosen, completion, total, selected, improves⟩
    · exact Or.inl unchanged
    · have chosenBound : chosen.sum ≤ bound := by simp only [Int.zero_add] at total; omega
      exact Or.inr ⟨chosen, completion.change_relation ordered agreement chosenBound, total, selected, improves⟩
  · intro chosen completion
    by_cases already : after.bound ≤ 0 + (chosen.sum : Int)
    · exact already
    · have chosenBound : chosen.sum ≤ bound := by omega
      exact optimal.minimal chosen (completion.change_relation ordered agreement.symm chosenBound)
  · intro chosen completion total improves
    have chosenBound : chosen.sum ≤ bound := by simp only [Int.zero_add] at total; omega
    exact optimal.first chosen (completion.change_relation ordered agreement.symm chosenBound) total improves

/-- The entire finite domain can be used when all candidate vertices are
small, including when no completion improves the sentinel. -/
theorem Optimal.change_relation_on {left right : Nat → Nat → Bool} {need : Nat} {values : List Nat}
    {before after : Best} {bound : Nat} (optimal : Optimal left need [] values 0 before after)
    (ordered : values.Pairwise (· < ·)) (agreement : AgreeBelow left right values bound)
    (bounded : ∀ p ∈ values, p ≤ bound) : Optimal right need [] values 0 before after := by
  have transfer {source target : Nat → Nat → Bool} (agree : AgreeBelow source target values bound)
      (chosen : List Nat) (completion : Completion source need values chosen) : Completion target need values chosen := by
    refine ⟨completion.size, completion.sublist, ?_⟩
    have sorted := List.Pairwise.sublist completion.sublist ordered
    apply (completion.compatible.and sorted).imp_of_mem
    intro p q memberP memberQ pair
    have inP := completion.sublist.mem memberP
    have inQ := completion.sublist.mem memberQ
    rw [← agree p inP q inQ pair.2 (bounded q inQ)]
    exact pair.1
  constructor
  · rcases optimal.sound with unchanged | ⟨chosen, completion, total, selected, improves⟩
    · exact Or.inl unchanged
    · exact Or.inr ⟨chosen, transfer agreement chosen completion, total, selected, improves⟩
  · intro chosen completion
    exact optimal.minimal chosen (transfer agreement.symm chosen completion)
  · intro chosen completion total improves
    exact optimal.first chosen (transfer agreement.symm chosen completion) total improves

/-- A certified improving optimum can be reused with any larger incumbent.
This permits finite checking with a tight initial bound, while the source
keeps its original sentinel and strict update behavior. -/
theorem Optimal.rebase {relation : Nat → Nat → Bool} {need : Nat} {clique values : List Nat}
    {sum : Int} {before after replacement : Best} (optimal : Optimal relation need clique values sum before after)
    (improves : after.bound < before.bound) (replacementBound : after.bound < replacement.bound) :
    Optimal relation need clique values sum replacement after := by
  constructor
  · rcases optimal.sound with same | ⟨chosen, completion, total, selected, _⟩
    · rw [same] at improves
      omega
    · exact Or.inr ⟨chosen, completion, total, selected, replacementBound⟩
  · exact optimal.minimal
  · intro chosen completion total _
    exact optimal.first chosen completion total improves

end ProjectEulerSolutions.CliqueSearch
