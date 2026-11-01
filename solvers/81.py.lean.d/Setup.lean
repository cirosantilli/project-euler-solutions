import solvers.«81.py.lean.d».Outer

namespace solvers.«81»
open FormalSW.PyAst HeapCalls ProjectEulerStatements.P81
open ProjectEulerSolutions.RectangularMatrixCalls
open ProjectEulerSolutions.TriangleBufferCalls (working number found prefix_found)
open ProjectEulerSolutions.GridPathBufferCalls
set_option Elab.async false

abbrev scratch (heap : Mutable.Heap) : Mutable.Heap := heap ++ [.list [number 0]]
abbrev firstFor : Stmt :=
  .forIn (.name "j") (.call (.name "range") [.constant (.int 1), .name "m"]) firstBody []
abbrev outerFor : Stmt :=
  .forIn (.name "i") (.call (.name "range") [.constant (.int 1), .name "n"]) matrixBody []
abbrev setupTail : List Stmt :=
  [.assign [.name "row"] (.subscript (.name "matrix") (.constant (.int 0))),
   .assign [.name "dp"] (.binOp (.list [.constant (.int 0)]) .mult (.name "m"))] ++
    initialColumn ++ [firstFor, outerFor, finalReturn]

theorem allocate_buffer (width : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getWidth : env "m" = some (number width)) :
    Evaluates program env (.binOp (.list [.constant (.int 0)]) .mult (.name "m")) heap
      (.ref (scratch heap).length) (working (scratch heap) (List.replicate width 0)) := by
  apply Evaluates.binary (av := .ref heap.length) (bv := number width)
    (middle := scratch heap) (afterRight := scratch heap)
  · exact Evaluates.list (.cons Evaluates.constant .nil)
  · exact Evaluates.lookup getWidth
  · simp [Mutable.binary, show (BinOp.mult == BinOp.mult) = true from rfl,
      Mutable.array, Mutable.allocate, number, scratch, working,
      ProjectEulerSolutions.TriangleBufferCalls.encode, Except.map]

def answerBuffer (first : Nat × List Int) (rest : MatrixRows) : List Int :=
  (rest.map Prod.snd).foldl rowStep (firstRow first.2)

theorem positive_body (heap : Mutable.Heap) (env : HeapCalls.Env) (address : Nat)
    (first : Nat × List Int) (rest : MatrixRows)
    (getMatrix : heap[address]? = some (.list ((first :: rest).map (fun pair => .ref pair.1))))
    (getRows : ∀ pair ∈ first :: rest, heap[pair.1]? = some (.list (encode pair.2)))
    (shape : ∀ pair ∈ first :: rest, pair.2.length = first.2.length)
    (positive : 0 < first.2.length)
    (localMatrix : env "matrix" = some (.ref address))
    (localWidth : env "m" = some (number first.2.length))
    (localHeight : env "n" = some (number (rest.length + 1))) :
    BlockReturns program env setupTail heap
      (number (ProjectEulerStatements.P81.naive ((first :: rest).map Prod.snd)))
      (working (scratch heap) (answerBuffer first rest)) := by
  let entered := setLocal env "row" (.ref first.1)
  let ready := setLocal entered "dp" (.ref (scratch heap).length)
  have readFirst : Evaluates program env (.subscript (.name "matrix") (.constant (.int 0))) heap (.ref first.1) heap := by
    apply Evaluates.subscript (by intros a b c impossible; cases impossible) (Evaluates.lookup localMatrix) Evaluates.constant
    simp [Mutable.getItem, getMatrix, Mutable.index, Except.map]
  have savedFirst : (scratch heap)[first.1]? = some (.list (encode first.2)) :=
    prefix_found heap [0] first.1 _ (getRows first (by simp))
  have initial := initial_column_result program (scratch heap) ready first.1 first.2 savedFirst positive
    (by simp [ready, entered, setLocal]) rfl
  obtain ⟨after, loop, kept⟩ := first_loop program (scratch heap) ready first.1 1 first.2 savedFirst
    (by omega) (by omega) (by simp [ready, entered, setLocal]) rfl
  have firstBlock : BlockContinues program ready [firstFor]
      (working (scratch heap) (partialBuffer (firstRow first.2) (List.replicate first.2.length 0) 1)) after
      (working (scratch heap) (firstRow first.2)) := by
    apply BlockContinues.for_range (.cons Evaluates.constant
      (.cons (Evaluates.lookup (by simpa [ready, entered, setLocal] using localWidth)) .nil)) rfl
    exact loop
  have outer := matrix_loop 1 first.2.length (scratch heap) after address (first :: rest) (firstRow first.2)
    (prefix_found heap [0] address _ getMatrix)
    (fun pair member => prefix_found heap [0] pair.1 _ (getRows pair member)) shape (by simp) positive
    (by simp [firstRow])
    (by rw [kept "matrix" (by decide)]; simpa [ready, entered, setLocal] using localMatrix)
    (by rw [kept "dp" (by decide)]; simp [ready, setLocal])
    (by rw [kept "m" (by decide)]; simpa [ready, entered, setLocal] using localWidth)
  have finish : BlockReturns program after [outerFor, finalReturn]
      (working (scratch heap) (firstRow first.2))
      (number (ProjectEulerStatements.P81.naive ((first :: rest).map Prod.snd)))
      (working (scratch heap) (answerBuffer first rest)) := by
    apply BlockReturns.for_range (.cons Evaluates.constant
      (.cons (Evaluates.lookup (by rw [kept "n" (by decide)]; simpa [ready, entered, setLocal] using localHeight)) .nil)) rfl
    simpa only [remainingBuffer, List.drop_succ_cons, List.drop_zero, List.map_cons,
      ProjectEulerStatements.P81.naive, minPath, answerBuffer, List.length_cons] using outer
  apply BlockReturns.assign readFirst
  apply BlockReturns.assign (allocate_buffer first.2.length entered heap (by simpa [entered, setLocal] using localWidth))
  exact (initial.append firstBlock).returns finish

end solvers.«81»
