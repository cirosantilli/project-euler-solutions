import ProjectEulerSolutions.Lib.RationalSetStorage
import ProjectEulerSolutions.Lib.RationalChoices
import ProjectEulerSolutions.Lib.ArithmeticReduction
import ProjectEulerSolutions.Lib.HeapPreservation

/-! Complete canonical result sets for arithmetic reduction states. -/
namespace ProjectEulerSolutions.RationalPair
open FormalSW.PyAst ArithmeticExpressions
set_option Elab.async false

theorem Canonical.value_injective {left right : Pair} (a : Canonical left) (b : Canonical right)
    (equal : value left = value right) : left = right := by
  calc
    left = ((value left).num, ((value left).den : Int)) := a
    _ = ((value right).num, ((value right).den : Int)) := by rw [equal]
    _ = right := b.symm

def Expected (state : List Pair) (pair : Pair) : Prop := Canonical pair ∧ Reduces (state.map value) (value pair)

theorem expected_empty (pair : Pair) : ¬ Expected [] pair := by
  rintro ⟨_, derived⟩
  obtain ⟨ordered, permutation, tree⟩ := derived.sound
  have empty : ordered = [] := List.perm_nil.mp permutation
  exact tree.nonempty empty

theorem expected_singleton (pair candidate : Pair) (canonical : Canonical pair) :
    Expected [pair] candidate ↔ candidate = pair := by
  constructor
  · rintro ⟨candidateCanonical, derived⟩
    obtain ⟨ordered, permutation, tree⟩ := derived.sound
    have same : ordered = [value pair] := List.perm_singleton.mp permutation
    rw [same] at tree
    exact candidateCanonical.value_injective canonical tree.singleton
  · rintro rfl
    exact ⟨canonical, Reduces.leaf _⟩

theorem expected_perm {left right : List Pair} (permutation : left.Perm right) (pair : Pair) :
    Expected left pair ↔ Expected right pair := by
  constructor
  · rintro ⟨canonical, derived⟩
    exact ⟨canonical, derived.perm (permutation.map value)⟩
  · rintro ⟨canonical, derived⟩
    exact ⟨canonical, derived.perm (permutation.symm.map value)⟩

namespace Heap

def ResultAt (heap : Mutable.Heap) (address : Nat) (state : List Pair) : Prop :=
  ∃ entries, SetAt heap address entries ∧ ∀ pair, pair ∈ entries.map Prod.fst ↔ Expected state pair

theorem ResultAt.bound {heap : Mutable.Heap} {address : Nat} {state : List Pair}
    (result : ResultAt heap address state) : address < heap.length := by
  obtain ⟨entries, found, _⟩ := result
  exact found.bound

theorem ResultAt.empty {heap : Mutable.Heap} {address : Nat} (found : heap[address]? = some (.set [])) :
    ResultAt heap address [] := by
  refine ⟨[], ⟨.inr ⟨rfl, found⟩, ?_⟩, ?_⟩
  · intro entry member
    simp at member
  · intro pair
    simp [expected_empty]

theorem ResultAt.singleton {heap : Mutable.Heap} {address operand : Nat} {pair : Pair}
    (found : heap[address]? = some (.keySet [(key pair, .ref operand)]))
    (row : heap[operand]? = some (object pair)) (canonical : Canonical pair) :
    ResultAt heap address [pair] := by
  refine ⟨[(pair, operand)], ⟨.inl found, ?_⟩, ?_⟩
  · intro entry member
    obtain rfl := List.mem_singleton.mp member
    exact row
  · intro candidate
    simp [expected_singleton pair candidate canonical]

theorem SetAt.preserved {memo address : Nat} {before after : Mutable.Heap} {pairs : List SetEntry}
    (found : SetAt before address pairs) (frame : PreservesExcept memo before after)
    (entries : List (HashDict.Entry Mutable.Value))
    (dictionary : (before[memo]?).bind Mutable.dictionaryEntries = some entries) : SetAt after address pairs := by
  constructor
  · rcases found.stored with stored | ⟨empty, stored⟩
    · exact .inl (frame.non_dictionary entries dictionary address _ stored rfl)
    · exact .inr ⟨empty, frame.non_dictionary entries dictionary address _ stored rfl⟩
  · intro entry member
    exact frame.non_dictionary entries dictionary entry.2 _ (found.rows entry member) rfl

theorem ResultAt.preserved {memo address : Nat} {before after : Mutable.Heap} {state : List Pair}
    (result : ResultAt before address state) (frame : PreservesExcept memo before after)
    (entries : List (HashDict.Entry Mutable.Value))
    (dictionary : (before[memo]?).bind Mutable.dictionaryEntries = some entries) : ResultAt after address state := by
  obtain ⟨pairs, found, members⟩ := result
  exact ⟨pairs, found.preserved frame entries dictionary, members⟩

theorem ResultAt.append {heap : Mutable.Heap} {address : Nat} {state : List Pair}
    (result : ResultAt heap address state) (extra : Mutable.Heap) : ResultAt (heap ++ extra) address state := by
  obtain ⟨pairs, found, members⟩ := result
  exact ⟨pairs, found.append extra, members⟩

end Heap
end ProjectEulerSolutions.RationalPair
