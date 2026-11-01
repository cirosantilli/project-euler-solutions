import ProjectEulerSolutions.Lib.HeapPreservation

/-! A mutation confined to fresh storage preserves the caller's entire heap prefix. -/
namespace ProjectEulerSolutions.PreservesExcept
open FormalSW.PyAst
set_option Elab.async false

theorem prefix_of_fresh {address : Nat} {before after : Mutable.Heap}
    (frame : PreservesExcept address before after) (fresh : before.length ≤ address) :
    before <+: after := by
  rw [List.prefix_iff_getElem?]
  intro index inside
  exact frame.objects index before[index] (by omega) (List.getElem?_eq_getElem inside)

end ProjectEulerSolutions.PreservesExcept
