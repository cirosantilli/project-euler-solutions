import ProjectEulerSolutions.Lib.CliqueSearchAgreement

namespace ProjectEulerSolutions.CliqueSearch

/-- Every actual edge is retained; extra possible edges are permitted. -/
def IncludedOn (actual possible : Nat → Nat → Bool) (values : List Nat) : Prop :=
  ∀ p ∈ values, ∀ q ∈ values, p < q → actual p q = true → possible p q = true

theorem Completion.include_relation {actual possible : Nat → Nat → Bool} {values chosen : List Nat} {need : Nat}
    (completion : Completion actual need values chosen) (ordered : values.Pairwise (· < ·))
    (included : IncludedOn actual possible values) : Completion possible need values chosen := by
  refine ⟨completion.size, completion.sublist, ?_⟩
  have sorted := List.Pairwise.sublist completion.sublist ordered
  apply (completion.compatible.and sorted).imp_of_mem
  intro p q memberP memberQ pair
  exact included p (completion.sublist.mem memberP) q (completion.sublist.mem memberQ) pair.2 pair.1

/-- A possible-edge search proves minimum and tie order for the actual
relation once its selected result has been checked separately. -/
theorem Optimal.of_supergraph {actual possible : Nat → Nat → Bool} {need : Nat}
    {clique values : List Nat} {sum : Int} {before after : Best}
    (optimal : Optimal possible need clique values sum before after)
    (ordered : values.Pairwise (· < ·)) (included : IncludedOn actual possible values)
    (sound : Sound actual need clique values sum before after) :
    Optimal actual need clique values sum before after := by
  refine ⟨sound, ?_, ?_⟩
  · intro chosen completion
    exact optimal.minimal chosen (completion.include_relation ordered included)
  · intro chosen completion total improves
    exact optimal.first chosen (completion.include_relation ordered included) total improves

theorem search_eq_of_supergraph {actual possible : Nat → Nat → Bool} {need : Nat}
    {clique values : List Nat} {sum : Int} {before after : Best}
    (ordered : values.Pairwise (· < ·)) (included : IncludedOn actual possible values)
    (checked : search possible need clique values sum before = after)
    (sound : Sound actual need clique values sum before after) :
    search actual need clique values sum before = after := by
  have optimal := search_optimal possible need clique values sum before ordered
  rw [checked] at optimal
  exact search_eq_of_optimal actual need clique values sum before after ordered
    (optimal.of_supergraph ordered included sound)

end ProjectEulerSolutions.CliqueSearch
