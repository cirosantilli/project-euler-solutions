import ProjectEulerSolutions.Lib.RationalResultSpec

/-! Updating a private collector preserves complete result sets at other addresses. -/
namespace ProjectEulerSolutions.RationalPair.Heap
open FormalSW.PyAst
set_option Elab.async false

theorem SetAt.no_dictionary {heap : Mutable.Heap} {address : Nat} {entries : List SetEntry}
    (found : SetAt heap address entries) : (heap[address]?).bind Mutable.dictionaryEntries = none := by
  rcases found.stored with stored | ⟨_, stored⟩ <;> simp [stored, setObject, Mutable.dictionaryEntries]

theorem SetAt.preserved_away {before after : Mutable.Heap} {address source : Nat} {target entries : List SetEntry}
    (found : SetAt before source entries) (collector : SetAt before address target)
    (frame : PreservesExcept address before after) (different : source ≠ address) : SetAt after source entries := by
  constructor
  · rcases found.stored with stored | ⟨empty, stored⟩
    · exact .inl (frame.objects source _ different stored)
    · exact .inr ⟨empty, frame.objects source _ different stored⟩
  · intro entry member
    have row := found.rows entry member
    apply frame.objects entry.2 _ _ row
    intro equal
    rcases collector.stored with stored | ⟨_, stored⟩ <;>
      simp [← equal, row, object, setObject] at stored

theorem ResultAt.preserved_away {before after : Mutable.Heap} {address source : Nat}
    {target : List SetEntry} {state : List Pair} (result : ResultAt before source state)
    (collector : SetAt before address target) (frame : PreservesExcept address before after)
    (different : source ≠ address) : ResultAt after source state := by
  obtain ⟨entries, found, members⟩ := result
  exact ⟨entries, found.preserved_away collector frame different, members⟩

end ProjectEulerSolutions.RationalPair.Heap
