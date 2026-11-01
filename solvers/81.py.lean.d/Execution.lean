import solvers.«81.py.lean.d».Setup

namespace solvers.«81»
open FormalSW.PyAst HeapCalls ProjectEulerStatements.P81
open ProjectEulerSolutions.RectangularMatrixCalls
open ProjectEulerSolutions.TriangleBufferCalls (working number)
set_option Elab.async false

private theorem fold_empty (rows : List (List Int)) : rows.foldl rowStep [] = [] := by
  induction rows with
  | nil => rfl
  | cons row rows ih => cases row <;> simpa [rowStep] using ih

theorem solve.result (heap : Mutable.Heap) (address : Nat) (rows : MatrixRows)
    (getMatrix : heap[address]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (getRows : ∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2))) (shape : Shape rows)
    (checked : ReturnsValue program "is_rectangular" [.ref address] heap (.scalar (.bool true)) heap) :
    ∃ extra, ReturnsValue program "solve" [.ref address] heap
      (number (ProjectEulerStatements.P81.naive (rows.map Prod.snd))) (heap ++ extra) := by
  let entry := HeapCalls.argumentEnv arguments [.ref address]
  have tested : Evaluates program entry (.unaryOp .not (.name "matrix")) heap
      (.scalar (.bool (decide (rows = [])))) heap :=
    Evaluates.unary (Evaluates.lookup rfl)
      (by simp [HeapCalls.unary, Mutable.truth, getMatrix]; cases rows <;> rfl)
  cases rows with
  | nil =>
      refine ⟨[], ?_⟩
      apply BlockReturns.function (args := arguments) rfl rfl
      apply BlockReturns.branch (flag := true) tested rfl
      simpa [ProjectEulerStatements.P81.naive, minPath, number] using
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
        have empty : first.2 = [] := by simpa using emptyWidth
        simpa [ProjectEulerStatements.P81.naive, minPath, empty, firstRow, firstRowFrom, fold_empty, number, sized, emptyWidth] using
          (BlockReturns.return_value (module := program) (env := sized) (heap := heap) Evaluates.constant
            : BlockReturns program sized (.return (.constant (.int 0)) :: _) heap (number 0) heap)
      · refine ⟨[.list [number 0], .list (encode (answerBuffer first rest))], ?_⟩
        apply BlockReturns.function (args := arguments) rfl rfl
        apply BlockReturns.branch (flag := false) tested rfl
        apply BlockReturns.assertion (Evaluates.call (.cons (Evaluates.lookup rfl) .nil) checked) rfl
        apply BlockReturns.assign height
        apply BlockReturns.assign width
        apply BlockReturns.branch (flag := false) (by simpa only [emptyWidth, decide_false] using testWidth) rfl
        simpa only [working, scratch, List.append_assoc, List.cons_append, List.nil_append] using
          positive_body heap sized address first rest getMatrix getRows rectangular (by omega) rfl rfl rfl

end solvers.«81»
