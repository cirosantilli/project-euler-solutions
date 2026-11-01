import solvers.«70.py.lean.d».ScanWidth

/-! Conditional screens and the cached record update reach the same next candidate. -/
namespace solvers.«70».scan
open FormalSW.PyAst HeapCalls ProjectEulerStatements TotientPermutation
set_option Elab.async false

abbrev updateCode : List Stmt :=
  [.assign [.name "best_n"] (.name "n"), .assign [.name "best_phi"] (.name "pn")]

theorem update_result (limit n best bestPhi width address : Nat) (heap : Mutable.Heap)
    (env : HeapCalls.Env) (ready : Ready limit n best bestPhi width address env)
    (getPN : env "pn" = some (number n.totient)) :
    ∃ finalEnv, BlockContinues program env updateCode heap finalEnv heap ∧
      Ready limit n n n.totient width address finalEnv := by
  let first := setLocal env "best_n" (number n)
  let finalEnv := setLocal first "best_phi" (number n.totient)
  refine ⟨finalEnv, ?_, ?_⟩
  · apply (BlockContinues.assign (Evaluates.lookup ready.getN)).append
    exact BlockContinues.assign (Evaluates.lookup (by simpa [first, setLocal] using getPN))
  · constructor
    · simpa [finalEnv, first, setLocal] using ready.getLimit
    · simpa [finalEnv, first, setLocal] using ready.getN
    · simp [finalEnv, first, setLocal]
    · simp [finalEnv, first, setLocal]
    · simpa [finalEnv, first, setLocal] using ready.getLow
    · simpa [finalEnv, first, setLocal] using ready.getHigh
    · simpa [finalEnv, first, setLocal] using ready.getPhi

/-- All screens and both record-update branches use the reference visit before continuing. -/
theorem tail_result (limit n best bestPhi width address : Nat) (heap : Mutable.Heap)
    (env : HeapCalls.Env) (ready : Ready limit n best bestPhi width address env)
    (getPN : env "pn" = some (number n.totient)) (found : PhiTable limit address heap)
    (large : 2 ≤ n) (lower : 10 ^ width ≤ n) (upper : n < 10 ^ (width + 1))
    (matched : PhiMatches best bestPhi) (next : Stmt) (following : List Stmt)
    (answer : Nat → Nat)
    (continuation : ∀ nextBest nextPhi nextEnv nextHeap,
      Ready limit n nextBest nextPhi width address nextEnv → PhiMatches nextBest nextPhi →
      PhiTable limit address nextHeap →
      ∃ finalHeap, BlockReturns program nextEnv (next :: following) nextHeap
        (number (answer nextBest)) finalHeap) :
    ∃ finalHeap, BlockReturns program env
      (scanBody.drop 2 ++ [.loopResume next] ++ following) heap
      (number (answer (visit best n))) finalHeap := by
  have widthTest := width_condition_result n.totient (10 ^ width) (10 ^ (width + 1))
    heap env getPN ready.getLow ready.getHigh
  by_cases badWidth : n.totient < 10 ^ width ∨ 10 ^ (width + 1) ≤ n.totient
  · have absent : ¬ eligible n := by
      rintro ⟨_, same⟩
      obtain ⟨phiLower, phiUpper, _⟩ := filters n width same lower upper
      omega
    obtain ⟨finalHeap, continued⟩ := continuation best bestPhi env heap ready matched found
    refine ⟨finalHeap, ?_⟩
    simp only [visit, if_neg absent]
    rw [scan_conditions_source]
    apply BlockReturns.branch (flag := true) (value := flag true)
      (by simpa only [badWidth, decide_true] using widthTest) rfl
    exact BlockReturns.continue_loop rfl continued
  · have residueTest := residue_condition_result n n.totient heap env ready.getN getPN
    by_cases badResidue : ((n : Int) - n.totient) % 9 ≠ 0
    · have absent : ¬ eligible n := by
        rintro ⟨_, same⟩
        exact badResidue (permutation_difference n n.totient same)
      obtain ⟨finalHeap, continued⟩ := continuation best bestPhi env heap ready matched found
      refine ⟨finalHeap, ?_⟩
      simp only [visit, if_neg absent]
      rw [scan_conditions_source]
      apply BlockReturns.branch (flag := false) (value := flag false)
        (by simpa only [badWidth, decide_false] using widthTest) rfl
      apply BlockReturns.branch (flag := true) (value := flag true)
        (by simpa [badResidue] using residueTest) rfl
      exact BlockReturns.continue_loop rfl continued
    · let allocated := heap ++ signatureObjects n ++ signatureObjects n.totient
      have kept : PhiTable limit address allocated :=
        (found.append (signatureObjects n)).append (signatureObjects n.totient)
      have counted := signature_condition_result n n.totient heap env ready.getN getPN
      cases same : isPermutation n n.totient with
      | false =>
          have absent : ¬ eligible n := by rintro ⟨_, h⟩; simp [same] at h
          obtain ⟨finalHeap, continued⟩ :=
            continuation best bestPhi env allocated ready matched kept
          refine ⟨finalHeap, ?_⟩
          simp only [visit, if_neg absent]
          rw [scan_conditions_source]
          apply BlockReturns.branch (flag := false) (value := flag false)
            (by simpa only [badWidth, decide_false] using widthTest) rfl
          apply BlockReturns.branch (flag := false) (value := flag false)
            (by simpa [badResidue] using residueTest) rfl
          apply BlockReturns.branch (flag := true) (value := flag true)
            (by simpa only [same, Bool.not_false] using counted) rfl
          exact BlockReturns.continue_loop rfl continued
      | true =>
          have accepted : eligible n := ⟨by omega, same⟩
          have choice := improvement_matches n best bestPhi large matched
          have equivalent : (best = 0 ∨ n * bestPhi < best * n.totient) ↔
              best = 0 ∨ better n best = true := by
            simpa only [decide_eq_true_eq] using Bool.eq_iff_iff.mp choice
          have compared := improve_condition_result n n.totient best bestPhi allocated env
            ready.getN getPN ready.getBest ready.getBestPhi
          by_cases improved : best = 0 ∨ n * bestPhi < best * n.totient
          · have chosen := equivalent.mp improved
            obtain ⟨updatedEnv, updated, updatedReady⟩ :=
              update_result limit n best bestPhi width address allocated env ready getPN
            obtain ⟨finalHeap, continued⟩ := continuation n n.totient updatedEnv allocated
              updatedReady (fun _ => rfl) kept
            refine ⟨finalHeap, ?_⟩
            simp only [visit, if_pos accepted, if_pos chosen]
            rw [scan_conditions_source]
            apply BlockReturns.branch (flag := false) (value := flag false)
              (by simpa only [badWidth, decide_false] using widthTest) rfl
            apply BlockReturns.branch (flag := false) (value := flag false)
              (by simpa [badResidue] using residueTest) rfl
            apply BlockReturns.branch (flag := false) (value := flag false)
              (by simpa only [same, Bool.not_true] using counted) rfl
            apply BlockReturns.branch (flag := true) (value := flag true)
              (by simpa only [improved, decide_true] using compared) rfl
            exact updated.returns (BlockReturns.loop_resume continued)
          · have unchanged : ¬ (best = 0 ∨ better n best = true) :=
              fun h => improved (equivalent.mpr h)
            obtain ⟨finalHeap, continued⟩ :=
              continuation best bestPhi env allocated ready matched kept
            refine ⟨finalHeap, ?_⟩
            simp only [visit, if_pos accepted, if_neg unchanged]
            rw [scan_conditions_source]
            apply BlockReturns.branch (flag := false) (value := flag false)
              (by simpa only [badWidth, decide_false] using widthTest) rfl
            apply BlockReturns.branch (flag := false) (value := flag false)
              (by simpa [badResidue] using residueTest) rfl
            apply BlockReturns.branch (flag := false) (value := flag false)
              (by simpa only [same, Bool.not_true] using counted) rfl
            apply BlockReturns.branch (flag := false) (value := flag false)
              (by simpa only [improved, decide_false] using compared) rfl
            exact BlockReturns.loop_resume continued

end solvers.«70».scan
