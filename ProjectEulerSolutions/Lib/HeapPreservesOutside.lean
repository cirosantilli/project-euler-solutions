import ProjectEulerSolutions.Lib.HeapPreservation

/-! Loop bodies may update several private objects while preserving every other live object. -/
namespace ProjectEulerSolutions
open FormalSW.PyAst
set_option Elab.async false

structure PreservesOutside (addresses : List Nat) (before after : Mutable.Heap) : Prop where
  size : before.length ≤ after.length
  objects : ∀ other object, other ∉ addresses → before[other]? = some object → after[other]? = some object

namespace PreservesOutside

theorem refl (addresses : List Nat) (heap : Mutable.Heap) : PreservesOutside addresses heap heap :=
  ⟨Nat.le_refl _, fun _ _ _ found => found⟩

theorem trans {addresses : List Nat} {before middle after : Mutable.Heap}
    (first : PreservesOutside addresses before middle) (second : PreservesOutside addresses middle after) :
    PreservesOutside addresses before after :=
  ⟨Nat.le_trans first.size second.size, fun other object outside found =>
    second.objects other object outside (first.objects other object outside found)⟩

theorem of_except {address : Nat} {addresses : List Nat} {before after : Mutable.Heap}
    (frame : PreservesExcept address before after) (member : address ∈ addresses) : PreservesOutside addresses before after := by
  refine ⟨frame.size, ?_⟩
  intro other object outside found
  apply frame.objects other object _ found
  intro equal
  exact outside (equal ▸ member)

/-- A collector allocated outside the original caller's heap does not enlarge the caller's mutation footprint. -/
theorem fresh_collector {memo collector : Nat} {before middle after : Mutable.Heap}
    (frame : PreservesOutside [memo, collector] middle after) (caller : PreservesExcept memo before middle)
    (fresh : before.length ≤ collector) : PreservesExcept memo before after := by
  refine ⟨Nat.le_trans caller.size frame.size, ?_⟩
  intro other object different found
  have inside := (List.getElem?_eq_some_iff.mp found).1
  exact frame.objects other object (by simp; omega) (caller.objects other object different found)

/-- Dictionary and set writes cannot alias a live list or tuple object. -/
theorem ordinary {memo collector : Nat} {before after : Mutable.Heap}
    (frame : PreservesOutside [memo, collector] before after)
    (dictionary : List (HashDict.Entry Mutable.Value)) (keys : List (HashKey × Mutable.Value))
    (memoFound : (before[memo]?).bind Mutable.dictionaryEntries = some dictionary)
    (setFound : (before[collector]?).bind Mutable.keyEntries = some keys)
    (address : Nat) (object : Mutable.Object) (found : before[address]? = some object)
    (notDictionary : Mutable.dictionaryEntries object = none) (notSet : Mutable.keyEntries object = none) :
    after[address]? = some object := by
  have differentMemo : address ≠ memo := by
    intro equal
    simp [← equal, found, notDictionary] at memoFound
  have differentSet : address ≠ collector := by
    intro equal
    simp [← equal, found, notSet] at setFound
  exact frame.objects address object (by simp [differentMemo, differentSet]) found

end PreservesOutside
end ProjectEulerSolutions
