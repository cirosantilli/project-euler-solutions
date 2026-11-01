import ProjectEulerSolutions.Lib.HeapCollection

/-! Recursive calls may extend the heap and update one shared memo dictionary. -/
namespace ProjectEulerSolutions
open FormalSW.PyAst
set_option Elab.async false

structure PreservesExcept (address : Nat) (before after : Mutable.Heap) : Prop where
  size : before.length ≤ after.length
  objects : ∀ other object, other ≠ address → before[other]? = some object → after[other]? = some object

namespace PreservesExcept

theorem refl (address : Nat) (heap : Mutable.Heap) : PreservesExcept address heap heap :=
  ⟨Nat.le_refl _, fun _ _ _ found => found⟩

theorem trans {address : Nat} {before middle after : Mutable.Heap}
    (first : PreservesExcept address before middle) (second : PreservesExcept address middle after) :
    PreservesExcept address before after :=
  ⟨Nat.le_trans first.size second.size, fun other object different found =>
    second.objects other object different (first.objects other object different found)⟩

theorem append (address : Nat) (heap extra : Mutable.Heap) : PreservesExcept address heap (heap ++ extra) :=
  ⟨by simp, fun other object _ found => HeapCollection.prefix_found heap extra other object found⟩

theorem write (address : Nat) (heap : Mutable.Heap) (replacement : Mutable.Object) :
    PreservesExcept address heap (heap.set address replacement) := by
  refine ⟨by simp, ?_⟩
  intro other object different found
  simpa [List.getElem?_set, different, Ne.symm different] using found

/-- Writes to storage allocated after entry cannot affect any of the caller's objects. -/
theorem write_fresh {address : Nat} {before after : Mutable.Heap} (frame : PreservesExcept address before after)
    (fresh : Nat) (outside : before.length ≤ fresh) (replacement : Mutable.Object) :
    PreservesExcept address before (after.set fresh replacement) := by
  refine ⟨by simpa using frame.size, ?_⟩
  intro other object different found
  have inside := (List.getElem?_eq_some_iff.mp found).1
  have separate : fresh ≠ other := by omega
  simpa [List.getElem?_set, separate, Ne.symm separate] using frame.objects other object different found

theorem fresh_effects {address fresh : Nat} {before middle after : Mutable.Heap}
    (frame : PreservesExcept address before middle) (outside : before.length ≤ fresh)
    (size : middle.length ≤ after.length)
    (unchanged : ∀ other, other ≠ fresh → after[other]? = middle[other]?) : PreservesExcept address before after := by
  refine ⟨Nat.le_trans frame.size size, ?_⟩
  intro other object different found
  have inside := (List.getElem?_eq_some_iff.mp found).1
  rw [unchanged other (by omega)]
  exact frame.objects other object different found

/-- The memo dictionary cannot alias an existing object of another runtime type. -/
theorem non_dictionary {address : Nat} {before after : Mutable.Heap} (frame : PreservesExcept address before after)
    (entries : List (HashDict.Entry Mutable.Value))
    (dictionary : (before[address]?).bind Mutable.dictionaryEntries = some entries)
    (other : Nat) (object : Mutable.Object) (found : before[other]? = some object)
    (ordinary : Mutable.dictionaryEntries object = none) : after[other]? = some object := by
  apply frame.objects other object _ found
  intro equal
  subst other
  simp [found, ordinary] at dictionary

end PreservesExcept
end ProjectEulerSolutions
