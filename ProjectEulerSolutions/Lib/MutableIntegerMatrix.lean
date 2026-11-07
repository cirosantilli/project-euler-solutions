import ProjectEulerSolutions.Lib.HeapTables
import Mathlib.Data.Matrix.Basic

/-! Mutable nested integer lists with distinct row identities. -/
namespace ProjectEulerSolutions.MutableIntegerMatrix
open FormalSW.PyAst HeapCalls
set_option Elab.async false

variable {n m : Nat}

structure View (matrix : Matrix (Fin n) (Fin m) Int) (heap : Mutable.Heap)
    (root : Nat) (addresses : Fin n → Nat) : Prop where
  found : heap[root]? = some (.list (List.ofFn fun row => .ref (addresses row)))
  rows : ∀ row, heap[addresses row]? = some (.list (List.ofFn fun column => .scalar (.int (matrix row column))))
  distinct : Function.Injective addresses
  separate : ∀ row, addresses row ≠ root

theorem View.row {matrix : Matrix (Fin n) (Fin m) Int} {heap : Mutable.Heap}
    {root : Nat} {addresses : Fin n → Nat} (view : View matrix heap root addresses) (index : Fin n) :
    Mutable.getItem (.ref root) (.scalar (.int index.val)) heap =
      some (.ok (.ref (addresses index)), heap) := by
  simpa using Mutable.getItem_list heap root index.val _ view.found (by simp)

theorem View.entry {matrix : Matrix (Fin n) (Fin m) Int} {heap : Mutable.Heap}
    {root : Nat} {addresses : Fin n → Nat} (view : View matrix heap root addresses)
    (row : Fin n) (column : Fin m) :
    Mutable.getItem (.ref (addresses row)) (.scalar (.int column.val)) heap =
      some (.ok (.scalar (.int (matrix row column))), heap) := by
  simpa using Mutable.getItem_list heap (addresses row) column.val _ (view.rows row) (by simp)

theorem View.row_result {matrix : Matrix (Fin n) (Fin m) Int} {heap : Mutable.Heap}
    {root : Nat} {addresses : Fin n → Nat} (view : View matrix heap root addresses)
    (program : Module) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (receiver index : Expr) (row : Fin n)
    (getMatrix : Evaluates program env receiver heap (.ref root) heap globals)
    (getRow : Evaluates program env index heap (.scalar (.int row.val)) heap globals)
    (ordinary : ∀ a b c, index ≠ .slice a b c) :
    Evaluates program env (.subscript receiver index) heap (.ref (addresses row)) heap globals :=
  Evaluates.subscript ordinary getMatrix getRow (view.row row)

theorem View.entry_result {matrix : Matrix (Fin n) (Fin m) Int} {heap : Mutable.Heap}
    {root : Nat} {addresses : Fin n → Nat} (view : View matrix heap root addresses)
    (program : Module) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (receiver index : Expr) (row : Fin n) (column : Fin m)
    (getRow : Evaluates program env receiver heap (.ref (addresses row)) heap globals)
    (getColumn : Evaluates program env index heap (.scalar (.int column.val)) heap globals)
    (ordinary : ∀ a b c, index ≠ .slice a b c) :
    Evaluates program env (.subscript receiver index) heap (.scalar (.int (matrix row column))) heap globals :=
  Evaluates.subscript ordinary getRow getColumn (view.entry row column)

theorem ofFn_set {α : Type} (values : Fin n → α) (index : Fin n) (value : α) :
    (List.ofFn values).set index.val value = List.ofFn (Function.update values index value) := by
  apply List.ext_getElem
  · simp
  · intro position left right
    have inside : position < n := by simpa using right
    by_cases same : position = index.val
    · subst position
      simp
    · have different : (⟨position, inside⟩ : Fin n) ≠ index := by intro equal; exact same (congrArg Fin.val equal)
      simp [Ne.symm same, Function.update_of_ne different]

def write (heap : Mutable.Heap) (addresses : Fin n → Nat) (matrix : Matrix (Fin n) (Fin m) Int)
    (row : Fin n) (column : Fin m) (value : Int) : Mutable.Heap :=
  heap.set (addresses row) (.list (List.ofFn fun index =>
    .scalar (.int ((Function.update (matrix row) column value) index))))

theorem View.write_operation {matrix : Matrix (Fin n) (Fin m) Int} {heap : Mutable.Heap}
    {root : Nat} {addresses : Fin n → Nat} (view : View matrix heap root addresses)
    (row : Fin n) (column : Fin m) (value : Int) :
    Mutable.setItem (.ref (addresses row)) (.scalar (.int column.val)) (.scalar (.int value)) heap =
      some (.ok (.scalar .none), write heap addresses matrix row column value) := by
  rw [HeapTables.setItem_list heap (addresses row) column.val _ _ (view.rows row) (by simp), ofFn_set]
  congr 5
  funext index
  by_cases same : index = column
  · subst index
    simp
  · simp only [Function.update_of_ne same]

theorem View.write {matrix : Matrix (Fin n) (Fin m) Int} {heap : Mutable.Heap}
    {root : Nat} {addresses : Fin n → Nat} (view : View matrix heap root addresses)
    (row : Fin n) (column : Fin m) (value : Int) :
    View (Function.update matrix row (Function.update (matrix row) column value))
      (write heap addresses matrix row column value) root addresses := by
  refine ⟨?_, ?_, view.distinct, view.separate⟩
  · rw [MutableIntegerMatrix.write, List.getElem?_set_ne (view.separate row)]
    exact view.found
  · intro index
    by_cases same : index = row
    · subst index
      simp only [Function.update_self, MutableIntegerMatrix.write]
      exact List.getElem?_set_self (List.getElem?_eq_some_iff.mp (view.rows row)).1
    · rw [Function.update_of_ne same, MutableIntegerMatrix.write, List.getElem?_set_ne (fun equal => same (view.distinct equal.symm))]
      exact view.rows index

theorem View.reindex {matrix : Matrix (Fin n) (Fin m) Int} {heap : Mutable.Heap}
    {root : Nat} {addresses : Fin n → Nat} (view : View matrix heap root addresses) (rename : Equiv.Perm (Fin n)) :
    View (fun row column => matrix (rename row) column)
      (heap.set root (.list (List.ofFn fun row => .ref (addresses (rename row))))) root (addresses ∘ rename) := by
  refine ⟨List.getElem?_set_self (List.getElem?_eq_some_iff.mp view.found).1, ?_,
    view.distinct.comp rename.injective, fun row => view.separate (rename row)⟩
  intro row
  rw [Function.comp_apply, List.getElem?_set_ne (Ne.symm (view.separate (rename row)))]
  exact view.rows (rename row)

theorem swap_entries {α : Type} (values : Fin n → α) (first second : Fin n) :
    ((List.ofFn values).set second.val (values first)).set first.val (values second) =
      List.ofFn (fun index => values (Equiv.swap first second index)) := by
  rw [ofFn_set, ofFn_set]
  congr 1
  funext index
  by_cases atFirst : index = first
  · subst index
    simp
  · by_cases atSecond : index = second
    · subst index
      simp [atFirst]
    · simp [Function.update_of_ne atFirst, Function.update_of_ne atSecond,
        Equiv.swap_apply_of_ne_of_ne atFirst atSecond]

/-- The two pointer writes implement a permutation, including a pivot already on the diagonal. -/
theorem View.swap_operations {matrix : Matrix (Fin n) (Fin m) Int} {heap : Mutable.Heap}
    {root : Nat} {addresses : Fin n → Nat} (view : View matrix heap root addresses) (first second : Fin n) :
    let intermediate := heap.set root (.list ((List.ofFn fun row => Mutable.Value.ref (addresses row)).set second.val (.ref (addresses first))))
    Mutable.setItem (.ref root) (.scalar (.int second.val)) (.ref (addresses first)) heap =
      some (.ok (.scalar .none), intermediate) ∧
    Mutable.setItem (.ref root) (.scalar (.int first.val)) (.ref (addresses second)) intermediate =
      some (.ok (.scalar .none), heap.set root (.list (List.ofFn fun row => .ref (addresses (Equiv.swap first second row))))) := by
  dsimp only
  constructor
  · exact HeapTables.setItem_list _ _ _ _ _ view.found (by simp)
  · rw [HeapTables.setItem_list _ _ _ _ _
      (List.getElem?_set_self (List.getElem?_eq_some_iff.mp view.found).1) (by simp),
      swap_entries, List.set_set]

theorem write_preserves {matrix : Matrix (Fin n) (Fin m) Int} {heap : Mutable.Heap}
    {addresses : Fin n → Nat} (row : Fin n) (column : Fin m) (value : Int) (address : Nat)
    (different : addresses row ≠ address) :
    (write heap addresses matrix row column value)[address]? = heap[address]? := by
  exact List.getElem?_set_ne different

end ProjectEulerSolutions.MutableIntegerMatrix
