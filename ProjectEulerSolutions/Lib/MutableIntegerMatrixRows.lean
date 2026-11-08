import ProjectEulerSolutions.Lib.MutableIntegerMatrixAccess

/-! Replacing a matrix row with a separately allocated list preserves row identities. -/
namespace ProjectEulerSolutions.MutableIntegerMatrix
open FormalSW.PyAst HeapCalls
set_option Elab.async false
variable {n m : Nat}

def replaceRow (heap : Mutable.Heap) (root : Nat) (addresses : Fin n → Nat)
    (row : Fin n) (address : Nat) : Mutable.Heap :=
  heap.set root (.list (List.ofFn fun index => .ref (Function.update addresses row address index)))

theorem View.replaceRow_operation {matrix : Matrix (Fin n) (Fin m) Int} {heap : Mutable.Heap}
    {root : Nat} {addresses : Fin n → Nat} (view : View matrix heap root addresses)
    (row : Fin n) (address : Nat) :
    Mutable.setItem (.ref root) (.scalar (.int row.val)) (.ref address) heap =
      some (.ok (.scalar .none), replaceRow heap root addresses row address) := by
  rw [HeapTables.setItem_list heap root row.val _ _ view.found (by simp), ofFn_set]
  congr 5
  funext index
  by_cases same : index = row
  · subst index
    simp
  · simp [Function.update_of_ne same]

/-- The replaced row may be an existing row or a fresh list, provided it aliases no other row. -/
theorem View.replaceRow {matrix : Matrix (Fin n) (Fin m) Int} {heap : Mutable.Heap}
    {root : Nat} {addresses : Fin n → Nat} (view : View matrix heap root addresses)
    (row : Fin n) (address : Nat) (values : Fin m → Int)
    (found : heap[address]? = some (.list (List.ofFn fun column => .scalar (.int (values column)))))
    (notRoot : address ≠ root) (notRows : ∀ index, index ≠ row → address ≠ addresses index) :
    View (Function.update matrix row values) (MutableIntegerMatrix.replaceRow heap root addresses row address)
      root (Function.update addresses row address) := by
  refine ⟨List.getElem?_set_self (List.getElem?_eq_some_iff.mp view.found).1, ?_, ?_, ?_⟩
  · intro index
    by_cases same : index = row
    · subst index
      simp only [Function.update_self, MutableIntegerMatrix.replaceRow, List.getElem?_set_ne (Ne.symm notRoot)]
      exact found
    · simp only [Function.update_of_ne same, MutableIntegerMatrix.replaceRow,
        List.getElem?_set_ne (Ne.symm (view.separate index))]
      exact view.rows index
  · intro first second equal
    by_cases atFirst : first = row
    · subst first
      by_cases atSecond : second = row
      · exact atSecond.symm
      · simp only [Function.update_self, Function.update_of_ne atSecond] at equal
        exact False.elim (notRows second atSecond equal)
    · by_cases atSecond : second = row
      · subst second
        simp only [Function.update_self, Function.update_of_ne atFirst] at equal
        exact False.elim (notRows first atFirst equal.symm)
      · simp only [Function.update_of_ne atFirst, Function.update_of_ne atSecond] at equal
        exact view.distinct equal
  · intro index
    by_cases same : index = row
    · subst index
      simpa only [Function.update_self] using notRoot
    · simpa only [Function.update_of_ne same] using view.separate index

theorem replaceRow_preserves (heap : Mutable.Heap) (root : Nat) (addresses : Fin n → Nat)
    (row : Fin n) (address other : Nat) (different : root ≠ other) :
    (MutableIntegerMatrix.replaceRow heap root addresses row address)[other]? = heap[other]? :=
  List.getElem?_set_ne different

/-- A collector allocated at the old heap boundary cannot alias any existing row. -/
theorem View.replaceRow_fresh {matrix : Matrix (Fin n) (Fin m) Int} {heap : Mutable.Heap}
    {root : Nat} {addresses : Fin n → Nat} (view : View matrix heap root addresses)
    (row : Fin n) (values : Fin m → Int) (extra : Mutable.Heap)
    (found : (heap ++ extra)[heap.length]? =
      some (.list (List.ofFn fun column => .scalar (.int (values column))))) :
    View (Function.update matrix row values) (MutableIntegerMatrix.replaceRow (heap ++ extra) root addresses row heap.length)
      root (Function.update addresses row heap.length) := by
  apply (view.append extra).replaceRow row heap.length values found
  · have bound := (List.getElem?_eq_some_iff.mp view.found).1
    omega
  · intro index _
    have bound := (List.getElem?_eq_some_iff.mp (view.rows index)).1
    omega

/-- Evaluate a fresh row before writing its reference into the existing matrix. -/
theorem View.replaceRow_result {matrix : Matrix (Fin n) (Fin m) Int} {heap : Mutable.Heap}
    {root : Nat} {addresses : Fin n → Nat} (view : View matrix heap root addresses)
    (module : Module) (row : Fin n) (values : Fin m → Int) (extra : Mutable.Heap)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (expression : Expr)
    (getMatrix : env "matrix" = some (.ref root))
    (getRow : env "row" = some (.scalar (.int row.val)))
    (evaluated : Evaluates module env expression heap (.ref heap.length) (heap ++ extra) globals)
    (found : (heap ++ extra)[heap.length]? =
      some (.list (List.ofFn fun column => .scalar (.int (values column))))) :
    BlockContinues module env [.assign [.subscript (.name "matrix") (.name "row")] expression]
      heap env (MutableIntegerMatrix.replaceRow (heap ++ extra) root addresses row heap.length) globals ∧
    View (Function.update matrix row values) (MutableIntegerMatrix.replaceRow (heap ++ extra) root addresses row heap.length)
      root (Function.update addresses row heap.length) ∧
    ∀ address, address < heap.length → root ≠ address →
      (MutableIntegerMatrix.replaceRow (heap ++ extra) root addresses row heap.length)[address]? = heap[address]? := by
  refine ⟨?_, view.replaceRow_fresh row values extra found, ?_⟩
  · apply BlockContinues.setItem (by intros; simp) evaluated
      (.cons (.lookup getMatrix) (.cons (.lookup getRow) .nil))
    exact (view.append extra).replaceRow_operation row heap.length
  · intro address bounded different
    rw [replaceRow_preserves _ _ _ _ _ _ different, List.getElem?_append_left bounded]

end ProjectEulerSolutions.MutableIntegerMatrix
