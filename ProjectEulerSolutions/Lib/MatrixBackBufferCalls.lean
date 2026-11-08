import ProjectEulerSolutions.Lib.MutableIntegerMatrixAccess

/-! The private integer solution buffer used by both exact stationary solvers. -/
namespace ProjectEulerSolutions.MatrixBackBufferCalls
open FormalSW.PyAst HeapCalls
set_option Elab.async false

structure BackView (matrix : Matrix (Fin 120) (Fin 121) Int) (values : Fin 120 → Int)
    (heap : Mutable.Heap) (root : Nat) (addresses : Fin 120 → Nat) (weights : Nat) : Prop where
  matrixView : MutableIntegerMatrix.View matrix heap root addresses
  weightsFound : heap[weights]? = some (.list (List.ofFn fun index => .scalar (.int (values index))))
  notRoot : weights ≠ root
  notRows : ∀ row, weights ≠ addresses row

def weightsObjects : Mutable.Heap :=
  [.list [.scalar (.int 0)], .list (List.ofFn fun _ : Fin 120 => .scalar (.int 0))]

theorem weights_value (module : Module) (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getSize : env "size" = some (.scalar (.int 120))) :
    Evaluates module env (.binOp (.list [.constant (.int 0)]) .mult (.name "size"))
      heap (.ref (heap.length + 1)) (heap ++ weightsObjects) globals := by
  apply Evaluates.binary (Evaluates.list (.cons Evaluates.constant .nil)) (.lookup getSize)
  simp only [weightsObjects, List.ofFn_const]
  simp [Mutable.binary, Mutable.array, Mutable.allocate, Except.map, List.append_assoc,
    show (BinOp.mult == BinOp.mult) = true from rfl]

theorem weights_view (matrix : Matrix (Fin 120) (Fin 121) Int) (heap : Mutable.Heap)
    (root : Nat) (addresses : Fin 120 → Nat) (view : MutableIntegerMatrix.View matrix heap root addresses) :
    BackView matrix (fun _ => 0) (heap ++ weightsObjects) root addresses (heap.length + 1) := by
  refine ⟨view.append _, ?_, ?_, ?_⟩
  · rw [List.getElem?_append_right (show heap.length ≤ heap.length + 1 by omega)]
    simp only [Nat.add_sub_cancel_left, weightsObjects]
    rfl
  · have inside := (List.getElem?_eq_some_iff.mp view.found).1
    omega
  · intro row
    have inside := (List.getElem?_eq_some_iff.mp (view.rows row)).1
    omega

theorem BackView.read {matrix : Matrix (Fin 120) (Fin 121) Int} {values : Fin 120 → Int}
    {heap : Mutable.Heap} {root weights : Nat} {addresses : Fin 120 → Nat}
    (view : BackView matrix values heap root addresses weights) (index : Fin 120) :
    Mutable.getItem (.ref weights) (.scalar (.int index.val)) heap = some (.ok (.scalar (.int (values index))), heap) := by
  have read := Mutable.getItem_list heap weights index.val _ view.weightsFound
    (by simpa only [List.length_ofFn] using index.isLt)
  rw [List.getElem_ofFn] at read
  exact read

def writeWeight (heap : Mutable.Heap) (weights : Nat) (values : Fin 120 → Int) (index : Fin 120) (value : Int) : Mutable.Heap :=
  heap.set weights (.list (List.ofFn fun column => .scalar (.int (Function.update values index value column))))

theorem BackView.write_operation {matrix : Matrix (Fin 120) (Fin 121) Int} {values : Fin 120 → Int}
    {heap : Mutable.Heap} {root weights : Nat} {addresses : Fin 120 → Nat}
    (view : BackView matrix values heap root addresses weights) (index : Fin 120) (value : Int) :
    Mutable.setItem (.ref weights) (.scalar (.int index.val)) (.scalar (.int value)) heap =
      some (.ok (.scalar .none), writeWeight heap weights values index value) := by
  rw [HeapTables.setItem_list heap weights index.val _ _ view.weightsFound
    (by simpa only [List.length_ofFn] using index.isLt), MutableIntegerMatrix.ofFn_set]
  congr 5
  funext column
  by_cases same : column = index
  · subst column
    simp
  · simp only [Function.update_of_ne same]

theorem BackView.write {matrix : Matrix (Fin 120) (Fin 121) Int} {values : Fin 120 → Int}
    {heap : Mutable.Heap} {root weights : Nat} {addresses : Fin 120 → Nat}
    (view : BackView matrix values heap root addresses weights) (index : Fin 120) (value : Int) :
    BackView matrix (Function.update values index value) (writeWeight heap weights values index value) root addresses weights :=
  ⟨view.matrixView.set_other weights _ view.notRoot view.notRows,
    List.getElem?_set_self (List.getElem?_eq_some_iff.mp view.weightsFound).1, view.notRoot, view.notRows⟩

/-- The source writes a computed weight without changing any matrix row. -/
theorem BackView.write_result_in {matrix : Matrix (Fin 120) (Fin 121) Int} {values : Fin 120 → Int}
    {heap : Mutable.Heap} {root weights : Nat} {addresses : Fin 120 → Nat}
    (view : BackView matrix values heap root addresses weights) (module : Module) (index : Fin 120) (value : Int)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (expression : Expr)
    (evaluated : Evaluates module env expression heap (.scalar (.int value)) heap globals)
    (getWeights : env "weights" = some (.ref weights))
    (getRow : env "row" = some (.scalar (.int index.val))) :
    BlockContinues module env [.assign [.subscript (.name "weights") (.name "row")] expression]
      heap env (writeWeight heap weights values index value) globals :=
  BlockContinues.setItem (by intros; simp) evaluated (.cons (.lookup getWeights) (.cons (.lookup getRow) .nil))
    (view.write_operation index value)

end ProjectEulerSolutions.MatrixBackBufferCalls
