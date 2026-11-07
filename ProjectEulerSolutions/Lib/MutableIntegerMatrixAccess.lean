import ProjectEulerSolutions.Lib.MutableIntegerMatrix

namespace ProjectEulerSolutions.MutableIntegerMatrix
open FormalSW.PyAst
set_option Elab.async false

variable {n m : Nat}

theorem View.append {matrix : Matrix (Fin n) (Fin m) Int} {heap : Mutable.Heap}
    {root : Nat} {addresses : Fin n → Nat} (view : View matrix heap root addresses) (extra : Mutable.Heap) :
    View matrix (heap ++ extra) root addresses :=
  ⟨HeapCollection.prefix_found _ _ _ _ view.found, fun row => HeapCollection.prefix_found _ _ _ _ (view.rows row),
    view.distinct, view.separate⟩

theorem View.set_other {matrix : Matrix (Fin n) (Fin m) Int} {heap : Mutable.Heap}
    {root : Nat} {addresses : Fin n → Nat} (view : View matrix heap root addresses)
    (address : Nat) (object : Mutable.Object) (notRoot : address ≠ root) (notRows : ∀ row, address ≠ addresses row) :
    View matrix (heap.set address object) root addresses := by
  refine ⟨?_, ?_, view.distinct, view.separate⟩
  · rw [List.getElem?_set_ne notRoot]
    exact view.found
  · intro row
    rw [List.getElem?_set_ne (notRows row)]
    exact view.rows row

theorem getItem_fromEnd (heap : Mutable.Heap) (address : Nat) (values : Fin n → Mutable.Value)
    (found : heap[address]? = some (.list (List.ofFn values))) (index : Fin n) :
    Mutable.getItem (.ref address) (.scalar (.int (index.val - (n : Int)))) heap =
      some (.ok (values index), heap) := by
  have negative : (index.val : Int) - n < 0 := by have inside := index.isLt; omega
  have nonnegative : ¬(index.val : Int) < 0 := by omega
  have bounded : ¬(n : Int) ≤ index.val := by have inside := index.isLt; omega
  simp [Mutable.getItem, found, Mutable.index, List.length_ofFn, negative, nonnegative, bounded, Except.map]

theorem View.row_fromEnd {matrix : Matrix (Fin n) (Fin m) Int} {heap : Mutable.Heap}
    {root : Nat} {addresses : Fin n → Nat} (view : View matrix heap root addresses) (row : Fin n) :
    Mutable.getItem (.ref root) (.scalar (.int (row.val - (n : Int)))) heap =
      some (.ok (.ref (addresses row)), heap) := getItem_fromEnd heap root _ view.found row

theorem View.entry_fromEnd {matrix : Matrix (Fin n) (Fin m) Int} {heap : Mutable.Heap}
    {root : Nat} {addresses : Fin n → Nat} (view : View matrix heap root addresses) (row : Fin n) (column : Fin m) :
    Mutable.getItem (.ref (addresses row)) (.scalar (.int (column.val - (m : Int)))) heap =
      some (.ok (.scalar (.int (matrix row column))), heap) := getItem_fromEnd heap (addresses row) _ (view.rows row) column

end ProjectEulerSolutions.MutableIntegerMatrix
