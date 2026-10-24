import lang.py.FormalSW.PermutationCalls
import lang.py.FormalSW.SumCalls

/-! A generator sum acquires its permutation pool before evaluating the starting value. -/
namespace ProjectEulerSolutions.PermutationSums
open FormalSW.PyAst FormalSW.PyAst.HeapCalls

set_option Elab.async false

variable {globals : Mutable.Globals} {program : Module} {env : HeapCalls.Env}
  {expressions : List Expr} {values : List Mutable.Value} {heap middle afterStart finalHeap : Mutable.Heap}
  {rows : List (List SequenceSource)} {target element start : Expr}
  {filters : List Expr} {generators : List (Expr × Expr × List Expr)} {initial answer : Mutable.Value}

theorem evaluate
    (arguments : Evaluates.Arguments (globals := globals) program env expressions heap values middle)
    (captured : permutationRows values middle = some (.ok rows, middle))
    (started : Evaluates (globals := globals) program env start middle initial afterStart)
    (accepted : sumInitial initial afterStart = some (.ok initial, afterStart))
    (executed : BlockReturns (globals := globals) program
      (setLocal (comprehensionTargetsEnv env ((target, .permutationsIterable expressions, filters) :: generators))
        sumAccumulator initial)
      [.permutationStepLoop (comprehensionTargetName target) rows
        (comprehensionTargetBody target (comprehensionFilters filters
          (comprehensionTargetsBody (sumBody element []) generators))) [],
        .return (.name sumAccumulator)] afterStart answer finalHeap) :
    Evaluates (globals := globals) program env
      (.sumGeneratorTargets element ((target, .permutationsIterable expressions, filters) :: generators) start)
      heap answer finalHeap := by
  intro anchors stack
  apply Reaches.next rfl
  apply (Evaluates.arguments_reach arguments _ [] anchors stack).trans
  apply Reaches.next (following := ⟨.expr start env,
    .sumStart (.permutationStepLoop (comprehensionTargetName target) rows
      (comprehensionTargetBody target (comprehensionFilters filters
        (comprehensionTargetsBody (sumBody element []) generators))) [])
      (comprehensionTargetsEnv env ((target, .permutationsIterable expressions, filters) :: generators)) :: stack,
    middle⟩) (by simp [HeapCalls.step, sumGeneratorTargetsStart, HeapCalls.run, captured])
  apply (started anchors _).trans
  exact .next (by simp [HeapCalls.step, sumEnter, HeapCalls.run, accepted])
    ((executed anchors (.generatorEnd :: stack)).trans (.next rfl .refl))

end ProjectEulerSolutions.PermutationSums
