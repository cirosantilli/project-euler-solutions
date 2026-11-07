import solvers.«84.py.lean.d».StationaryResult
import ProjectEulerSolutions.Lib.MonopolyOutputFormatting

namespace solvers.«84».solve
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
set_option Elab.async false

def definition : List Arg × List Stmt :=
  (findFunctionIn program.body "solve").getD ([], [])

def denominator : Expr :=
  .binOp (.binOp (.constant (.int 256)) .mult (.name "dice_sides")) .mult (.name "dice_sides")

def weightsExpression : Expr := .call (.name "stationary_weights")
  [.call (.name "build_transition") [.name "dice_sides"], denominator]

theorem source : executableBody definition.2 =
    [.assign [.name "weights"] weightsExpression,
      .assign [.name "square_weights"] MonopolyOutput.squareCollection,
      .return (MonopolyOutput.outputExpression "pos")] := rfl

/-- The final two statements observe only the stationary weights and append private output objects. -/
theorem output_result (values : Fin 120 → Int) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (MonopolyOutput.encode values)))
    (env : HeapCalls.Env) (globals : Mutable.Globals) (getWeights : env "weights" = some (.ref address)) :
    ∃ extra, BlockReturns program env ((executableBody definition.2).drop 1) heap
      (.scalar (.string (MonopolyOutput.answer values))) (heap ++ extra) globals := by
  obtain ⟨garbage, collected⟩ := MonopolyOutput.squares_result program values heap address found env globals getWeights
  obtain ⟨extra, returned⟩ := MonopolyOutput.output_result program values
    (HeapCollection.working heap (MonopolyOutput.encode (Monopoly.squareWeight values)) garbage) heap.length
    (HeapCollection.found ..) (setLocal env "square_weights" (.ref heap.length)) globals "pos" (by decide) (by simp [setLocal])
  refine ⟨(.list (MonopolyOutput.encode (Monopoly.squareWeight values)) :: garbage) ++ extra, ?_⟩
  rw [source]
  simpa only [List.drop_succ_cons, List.drop_zero, HeapCollection.working, List.append_assoc] using
    (BlockContinues.assign collected).returns (BlockReturns.return_value returned)

end solvers.«84».solve
