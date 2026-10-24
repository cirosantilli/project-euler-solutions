import solvers.«70.scan.tail.source»

/-! The complete optimized range scan follows the same strict minimum fold. -/
namespace solvers.«70».scan
open FormalSW.PyAst HeapCalls ProjectEulerStatements TotientPermutation
set_option Elab.async false

structure Stable (limit best bestPhi width address : Nat) (env : HeapCalls.Env) : Prop where
  getLimit : env "limit" = some (number limit)
  getBest : env "best_n" = some (number best)
  getBestPhi : env "best_phi" = some (number bestPhi)
  getLow : env "low" = some (number (10 ^ width : Nat))
  getHigh : env "high" = some (number (10 ^ (width + 1) : Nat))
  getPhi : env "phi" = some (.ref address)

theorem Ready.stable {limit n best bestPhi width address : Nat} {env : HeapCalls.Env}
    (ready : Ready limit n best bestPhi width address env) :
    Stable limit best bestPhi width address env :=
  ⟨ready.getLimit, ready.getBest, ready.getBestPhi, ready.getLow, ready.getHigh, ready.getPhi⟩

theorem Stable.at_index {limit best bestPhi width address : Nat} {env : HeapCalls.Env}
    (stable : Stable limit best bestPhi width address env) (n : Nat) :
    Ready limit n best bestPhi width address (setLocal env "n" (number n)) where
  getLimit := by simpa [setLocal] using stable.getLimit
  getN := by simp [setLocal]
  getBest := by simpa [setLocal] using stable.getBest
  getBestPhi := by simpa [setLocal] using stable.getBestPhi
  getLow := by simpa [setLocal] using stable.getLow
  getHigh := by simpa [setLocal] using stable.getHigh
  getPhi := by simpa [setLocal] using stable.getPhi

abbrev pnCode : Stmt := .assign [.name "pn"] (.subscript (.name "phi") (.name "n"))

theorem scan_prefix_source : scanBody = [boundaryCode, pnCode] ++ scanBody.drop 2 := rfl

theorem scan_loop_continuation (next : Stmt) :
    Calls.loopContinuation scanBody next = .loopResume next := rfl

/-- The scan terminates over every finite bound and agrees with the reference update fold. -/
theorem loop_result (limit first count best bestPhi width address : Nat)
    (length : first + count = limit + 1) (large : 2 ≤ first) (before : WidthBefore first width)
    (matched : PhiMatches best bestPhi) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (stable : Stable limit best bestPhi width address env) (found : PhiTable limit address heap) :
    ∃ finalHeap, BlockReturns program env
      [.rangeStepLoop "n" first (limit + 1) 1 scanBody [], .return (.name "best_n")]
      heap (number ((List.range' first count).foldl visit best : Nat)) finalHeap := by
  induction count generalizing first best bestPhi width heap env with
  | zero =>
      refine ⟨heap, ?_⟩
      simp only [List.range'_zero, List.foldl_nil]
      apply BlockReturns.range_done (by simp [Calls.rangeContinues]; omega)
      exact BlockReturns.return_value (Evaluates.lookup stable.getBest)
  | succ count ih =>
      let indexed := setLocal env "n" (number first)
      have ready : Ready limit first best bestPhi width address indexed := stable.at_index first
      obtain ⟨widthEnv, widened, widthReady⟩ :=
        width_result limit first best bestPhi width address heap indexed ready
      let activeWidth := advanceWidth first width
      let loaded := setLocal widthEnv "pn" (number first.totient)
      have loadedReady : Ready limit first best bestPhi activeWidth address loaded :=
        widthReady.set_pn first.totient
      have assigned : BlockContinues program widthEnv [pnCode] heap loaded heap :=
        BlockContinues.assign
          (read_phi_result limit address first heap widthEnv found (by omega)
            widthReady.getPhi widthReady.getN)
      obtain ⟨lower, upper⟩ := advanceWidth_active first width before
      have nextBefore := advanceWidth_next first width before
      let next : Stmt := .rangeStepLoop "n" ((first : Int) + 1) (limit + 1) 1 scanBody []
      let answer : Nat → Nat := fun nextBest => (List.range' (first + 1) count).foldl visit nextBest
      have continuation : ∀ nextBest nextPhi nextEnv nextHeap,
          Ready limit first nextBest nextPhi activeWidth address nextEnv →
          PhiMatches nextBest nextPhi → PhiTable limit address nextHeap →
          ∃ finalHeap, BlockReturns program nextEnv [next, .return (.name "best_n")]
            nextHeap (number (answer nextBest)) finalHeap := by
        intro nextBest nextPhi nextEnv nextHeap nextReady nextMatched nextFound
        have continued := ih (first := first + 1) (best := nextBest) (bestPhi := nextPhi)
          (width := activeWidth) (length := by omega) (large := by omega) (before := nextBefore)
          (matched := nextMatched) (heap := nextHeap) (env := nextEnv)
          (stable := nextReady.stable) (found := nextFound)
        simpa only [next, answer, Int.natCast_add, Int.natCast_one] using continued
      obtain ⟨finalHeap, screened⟩ := tail_result limit first best bestPhi activeWidth address
        heap loaded loadedReady rfl found large lower upper matched next [.return (.name "best_n")]
        answer continuation
      refine ⟨finalHeap, ?_⟩
      apply BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
      rw [scan_loop_continuation]
      have joined := widened.returns (assigned.returns screened)
      have shape : scanBody ++ [.loopResume next] ++ [.return (.name "best_n")] =
          [boundaryCode, pnCode] ++
            (scanBody.drop 2 ++ [.loopResume next] ++ [.return (.name "best_n")]) := by
        conv_lhs => rw [scan_prefix_source]
        simp only [List.append_assoc, List.singleton_append]
      rw [shape]
      simpa only [answer, List.range'_succ, List.foldl_cons, List.append_assoc] using joined

end solvers.«70».scan
