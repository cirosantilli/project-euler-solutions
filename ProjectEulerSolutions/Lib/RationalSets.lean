import ProjectEulerSolutions.Lib.RationalKeys
import lang.py.FormalSW.ConstructionRelocation

/-! Shared rational tuple representatives in result sets. -/
namespace ProjectEulerSolutions.RationalPair.Heap
open FormalSW.PyAst Mutable.Relocation
set_option Elab.async false

/-- A singleton result keeps the operand tuple itself as its key representative. -/
theorem singleton_set (heap : Mutable.Heap) (address : Nat) (pair : Pair)
    (found : heap[address]? = some (object pair)) :
    Mutable.allocateSetValues [.ref address] heap =
      some (.ok (.ref heap.length), heap ++ [.keySet [(key pair, .ref address)]]) := by
  have row : Row heap (.ref address) pair := ⟨address, rfl, found⟩
  have hashed := (row.append [.set []]).hash
  rw [allocateSetValues_eq]
  simp [scalarValues, Mutable.allocateSet, Mutable.eval_bind, Mutable.addItem, hashed,
    Mutable.insertKey, Mutable.keyContains, List.set_append_right, Except.map]

end ProjectEulerSolutions.RationalPair.Heap
