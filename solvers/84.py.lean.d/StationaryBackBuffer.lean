import solvers.«84.py.lean.d».StationaryBackSetup

namespace solvers.«84».stationary_weights
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
set_option Elab.async false

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
theorem BackView.write_result {matrix : Matrix (Fin 120) (Fin 121) Int} {values : Fin 120 → Int}
    {heap : Mutable.Heap} {root weights : Nat} {addresses : Fin 120 → Nat}
    (view : BackView matrix values heap root addresses weights) (index : Fin 120) (value : Int)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (expression : Expr)
    (evaluated : Evaluates program env expression heap (.scalar (.int value)) heap globals)
    (getWeights : env "weights" = some (.ref weights))
    (getRow : env "row" = some (.scalar (.int index.val))) :
    BlockContinues program env [.assign [.subscript (.name "weights") (.name "row")] expression]
      heap env (writeWeight heap weights values index value) globals :=
  BlockContinues.setItem (by intros; simp) evaluated (.cons (.lookup getWeights) (.cons (.lookup getRow) .nil))
    (view.write_operation index value)

end solvers.«84».stationary_weights
