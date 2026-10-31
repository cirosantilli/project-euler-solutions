import solvers.«92.py.lean.d».WeightedLoop
import lang.py.FormalSW.DictionaryItemsSumCalls

namespace solvers.«92».summation
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.DigitSquareChains
set_option Elab.async false

abbrev program := Solvers.P92Ast.module
abbrev number := table.number
abbrev expression : Expr := .sumGeneratorTargets (.name "ways")
  [(bindings.target, .call (.attribute (.name "counts") "items") [], [weighted.condition])]
  (.constant (.int 0))

theorem source :
    (executableBody ((findFunctionIn program.body "solve").getD ([], [])).2)[4]? =
      some (.return expression) := weighted.source

/-- The source sum acquires the original dictionary and consumes its live cursor with a valid shared cache. -/
theorem result (heap : Mutable.Heap) (before : HeapCalls.Env) (globals : Mutable.Globals)
    (bound sourceAddress cache : Nat) (original : table.Entries) (cached : store.Entries)
    (counts : before "counts" = some (.ref sourceAddress))
    (known : before "known" = some (.ref cache))
    (sourceFound : table.Found heap sourceAddress original) (sourceKeys : table.Keys heap original)
    (found : store.Found heap cache cached) (valid : Memo.Valid (store.lookup cached))
    (different : sourceAddress ≠ cache) (exact : Prefix.Exact bound (table.view original)) :
    ∃ result finalHeap,
      Evaluates program before expression heap (number (count (bound + 1 : Nat))) finalHeap globals ∧
      store.Found finalHeap cache result ∧ Memo.Valid (store.lookup result) ∧
      rows.HeapFrame heap finalHeap cache := by
  let generatorLocals := setLocal (comprehensionTargetsEnv before
    [(bindings.target, .call (.attribute (.name "counts") "items") [], [weighted.condition])])
    sumAccumulator (number 0)
  obtain ⟨result, finalHeap, ran, finalFound, finalValid, frame⟩ :=
    weighted.result heap generatorLocals globals bound sourceAddress cache original cached
      (by simpa [generatorLocals, setLocal, comprehensionTargetsEnv, bindings.target, targetLocals, sumAccumulator] using known)
      (by simp [generatorLocals, setLocal]) sourceFound sourceKeys found valid different exact
  obtain ⟨object, present, represented⟩ := traversal.fields_found heap sourceAddress original sourceFound
  refine ⟨result, finalHeap, ?_, finalFound, finalValid, frame⟩
  apply Evaluates.sum_generator_items (Evaluates.lookup counts) present represented Evaluates.constant
    (by simp [sumInitial])
  simpa [generatorLocals, weighted.cursor, weighted.body, weighted.selected, weighted.addition,
    traversal.keys, traversal.fields, List.map_map, Function.comp_def, sequenceSource,
    comprehensionTargetName, comprehensionTargetBody, comprehensionFilters, comprehensionTargetsBody,
    bindings.target, bindings.body] using ran

end solvers.«92».summation
