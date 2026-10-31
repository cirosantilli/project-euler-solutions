import solvers.«93.py.lean.d».PairAllocation

namespace solvers.«93».pair_results
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.RationalPair
open ProjectEulerSolutions.HeapCollection
set_option Elab.async false

abbrev tail : List Stmt :=
  [.ifThen nb (appendBody forwardExpr) [], .ifThen na (appendBody backwardExpr) [], .return (.name "results")]

theorem source_tail : (executableBody ((findFunctionIn program.body "pair_results").getD ([], [])).2).drop 4 = tail := rfl

/-- Both conditional divisions preserve earlier results and all caller objects. -/
theorem finish_result (base : Mutable.Heap) (a b : Pair) (positiveA : 0 < a.2) (positiveB : 0 < b.2)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (ready : Ready env a b)
    (getOut : env "results" = some (.ref (listAddress base))) :
    ∃ extra, BlockReturns program env tail (initialHeap base a b) (.ref (listAddress base)) (base ++ extra) globals ∧
      Heap.ListFound (base ++ extra) (listAddress base) (choices a b) := by
  let buffer := base ++ firstObjects a b
  have length : buffer.length = listAddress base := by simp [buffer, initialChoices]
  have out : env "results" = some (.ref buffer.length) := by simpa only [length] using getOut
  obtain ⟨forwardRefs, forwardExtra, forwardRun, forwardRows⟩ :=
    conditional_append buffer (firstRefs base) [] (initialChoices a b) (division a b) b.1 env globals nb forwardExpr
      (initial_rows base a b) out (.lookup ready.numeratorB)
      (fun nonzero => forward a b positiveA nonzero ready)
  simp only [List.nil_append] at forwardRun forwardRows
  obtain ⟨backwardRefs, backwardExtra, backwardRun, backwardRows⟩ :=
    conditional_append buffer (firstRefs base ++ forwardRefs) forwardExtra
      (initialChoices a b ++ if b.1 = 0 then [] else [division a b]) (division b a) a.1
      env globals na backwardExpr forwardRows out (.lookup ready.numeratorA)
      (fun nonzero => backward a b positiveB nonzero ready)
  refine ⟨firstObjects a b ++ (.list ((firstRefs base ++ forwardRefs) ++ backwardRefs) ::
    (forwardExtra ++ backwardExtra)), ?_, ?_⟩
  · have returned := (forwardRun.append backwardRun).returns
      (BlockReturns.return_value (rest := []) (Evaluates.lookup getOut))
    simpa only [initialHeap, working, buffer, List.append_assoc] using returned
  · refine ⟨(firstRefs base ++ forwardRefs) ++ backwardRefs, ?_, ?_⟩
    · have found := ProjectEulerSolutions.HeapCollection.found buffer
        ((firstRefs base ++ forwardRefs) ++ backwardRefs) (forwardExtra ++ backwardExtra)
      simpa only [working, buffer, List.append_assoc, length] using found
    · simpa only [working, buffer, List.append_assoc, choices] using backwardRows

end solvers.«93».pair_results
