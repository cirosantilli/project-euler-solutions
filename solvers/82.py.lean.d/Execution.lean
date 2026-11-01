import solvers.«82.py.lean.d».Outer

namespace solvers.«82»
open FormalSW.PyAst HeapCalls ProjectEulerStatements.P82
open ProjectEulerSolutions.RectangularMatrixCalls
open ProjectEulerSolutions.TriangleBufferCalls (working number found prefix_found)
set_option Elab.async false

abbrev outerFor : Stmt := .forIn (.name "j")
  (.call (.name "range") [.constant (.int 1), .name "m"]) outerBody []

theorem columns_answer (first : Nat × List Int) (rest : MatrixRows) (positive : 0 < first.2.length) :
    minimum (remainingColumns (first :: rest) 1 first.2.length (column (first :: rest) 0)) =
      ProjectEulerStatements.P82.naive ((first :: rest).map Prod.snd) := by
  unfold ProjectEulerStatements.P82.naive minPath remainingColumns
  simp only [List.map_cons, List.headD_cons]
  cases width : first.2.length with
  | zero => omega
  | succ n =>
      simp [List.range_eq_range', List.range'_succ, List.foldl_map, getCol, Function.comp_def]

theorem solve.result (heap : Mutable.Heap) (address : Nat) (rows : MatrixRows)
    (getMatrix : heap[address]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (getRows : ∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2))) (shape : Shape rows)
    (checked : ReturnsValue program "is_rectangular" [.ref address] heap (.scalar (.bool true)) heap) :
    ∃ extra, ReturnsValue program "solve" [.ref address] heap
      (number (ProjectEulerStatements.P82.naive (rows.map Prod.snd))) (heap ++ extra) := by
  let entry := argumentEnv arguments [.ref address]
  have tested : Evaluates program entry (.unaryOp .not (.name "matrix")) heap
      (.scalar (.bool (decide (rows = [])))) heap :=
    Evaluates.unary (Evaluates.lookup rfl)
      (by simp [HeapCalls.unary, Mutable.truth, getMatrix]; cases rows <;> rfl)
  cases rows with
  | nil =>
      refine ⟨[], ?_⟩
      apply BlockReturns.function (args := arguments) rfl rfl
      apply BlockReturns.branch (flag := true) tested rfl
      simpa [ProjectEulerStatements.P82.naive, minPath, number] using
        (BlockReturns.return_value (module := program) (env := entry) (heap := heap) Evaluates.constant
          : BlockReturns program entry (.return (.constant (.int 0)) :: _) heap (number 0) heap)
  | cons first rest =>
      let tall := setLocal entry "n" (number (rest.length + 1))
      let sized := setLocal tall "m" (number first.2.length)
      have height : Evaluates program entry (.builtinCall .len [.name "matrix"])
          heap (number (rest.length + 1)) heap := by
        apply Evaluates.builtin (.cons (Evaluates.lookup rfl) .nil)
        simp [HeapCalls.primitive, Mutable.length, getMatrix]
      have readFirst : Evaluates program tall (.subscript (.name "matrix") (.constant (.int 0)))
          heap (.ref first.1) heap := by
        apply Evaluates.subscript (by intros a b c impossible; cases impossible) (Evaluates.lookup rfl) Evaluates.constant
        simp [Mutable.getItem, getMatrix, Mutable.index, Except.map]
      have width : Evaluates program tall (.builtinCall .len [.subscript (.name "matrix") (.constant (.int 0))])
          heap (number first.2.length) heap := by
        apply Evaluates.builtin (.cons readFirst .nil)
        simp [HeapCalls.primitive, Mutable.length, getRows first (by simp), encode]
      have testWidth : Evaluates program sized (.compare (.name "m") [(.eq, .constant (.int 0))])
          heap (.scalar (.bool (decide (first.2.length = 0)))) heap := by
        apply Evaluates.compare (av := number first.2.length) (bv := number 0)
          (middle := heap) (afterRight := heap) (Evaluates.lookup rfl) Evaluates.constant
        simp [HeapCalls.compare, scalarCompare, evalCmpOp]
      have rectangular : ∀ pair ∈ first :: rest, pair.2.length = first.2.length := by
        intro pair member
        obtain ⟨index, inside, rfl⟩ := List.mem_iff_getElem.mp member
        exact shape index inside
      by_cases emptyWidth : first.2.length = 0
      · refine ⟨[], ?_⟩
        apply BlockReturns.function (args := arguments) rfl rfl
        apply BlockReturns.branch (flag := false) tested rfl
        apply BlockReturns.assertion (Evaluates.call (.cons (Evaluates.lookup rfl) .nil) checked) rfl
        apply BlockReturns.assign height
        apply BlockReturns.assign width
        apply BlockReturns.branch (flag := true) (by simpa [sized, emptyWidth] using testWidth) rfl
        simpa [ProjectEulerStatements.P82.naive, minPath, emptyWidth, number, sized] using
          (BlockReturns.return_value (module := program) (env := sized) (heap := heap) Evaluates.constant
            : BlockReturns program sized (.return (.constant (.int 0)) :: _) heap (number 0) heap)
      · let initial := column (first :: rest) 0
        let base := working heap initial
        let ready := setLocal sized "dp" (.ref heap.length)
        have allocated := first_allocation heap sized address (first :: rest) getMatrix getRows
          (by intro pair member; rw [rectangular pair member]; omega) rfl rfl
        obtain ⟨extra, loop⟩ := columns_loop base ready address heap.length 1 first.2.length (first :: rest) initial
          (prefix_found heap initial address _ getMatrix)
          (fun pair member => prefix_found heap initial pair.1 _ (getRows pair member)) (found heap initial)
          rectangular (by simp [initial]) (by simp) (by omega) rfl rfl rfl
        rw [columns_answer first rest (by omega)] at loop
        refine ⟨[.list (encode initial)] ++ extra, ?_⟩
        apply BlockReturns.function (args := arguments) rfl rfl
        apply BlockReturns.branch (flag := false) tested rfl
        apply BlockReturns.assertion (Evaluates.call (.cons (Evaluates.lookup rfl) .nil) checked) rfl
        apply BlockReturns.assign height
        apply BlockReturns.assign width
        apply BlockReturns.branch (flag := false) (by simpa only [emptyWidth, decide_false] using testWidth) rfl
        apply BlockReturns.assign allocated
        apply BlockReturns.for_range (.cons Evaluates.constant (.cons (Evaluates.lookup rfl) .nil)) rfl
        simpa only [base, working, List.append_assoc] using loop

end solvers.«82»
