import solvers.«77.sieve.source»

/-! The optimized source search reaches the mathematically least answer for every signed threshold. -/
namespace solvers.«77».solve
open FormalSW.PyAst HeapCalls
open ProjectEulerStatements.PrimePartitions (count answer answer_spec answer_le)
set_option Elab.async false

private abbrev loopBody : List Stmt :=
  [.assign [.name "primes"] (.call (.name "sieve_primes") [.name "n"]),
    .assign [.name "cnt"] (.call (.name "count_prime_summations") [.name "n", .name "primes"]),
    .ifThen (.compare (.name "cnt") [(.gt, .name "target")]) [.return (.name "n")] [],
    .augAssign (.name "n") .add (.constant (.int 1))]
private abbrev statement : Stmt := .whileDo (.constant (.bool true)) loopBody []

private theorem loop (target : Int) (n : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getN : env "n" = some (number n)) (getTarget : env "target" = some (integer target))
    (lower : 2 ≤ n) (upper : n ≤ answer target) :
    ∃ extra, BlockReturns program env [statement] heap (number (answer target)) (heap ++ extra) := by
  obtain ⟨address, primeExtra, primes, found⟩ := sieve_primes.result (n : Int) heap
  obtain ⟨countExtra, counted⟩ := count_prime_summations.primes_append_result n (heap ++ primeExtra) address
    (by simpa only [Int.toNat_natCast] using found)
  let ready := setLocal (setLocal env "primes" (.ref address)) "cnt" (number (count n))
  have getN' : ready "n" = some (number n) := by simpa [ready, setLocal] using getN
  have getTarget' : ready "target" = some (integer target) := by simpa [ready, setLocal] using getTarget
  have check : Evaluates program ready (.compare (.name "cnt") [(.gt, .name "target")])
      ((heap ++ primeExtra) ++ countExtra) (.scalar (.bool (decide (target < (count n : Int)))))
      ((heap ++ primeExtra) ++ countExtra) :=
    Evaluates.compare (av := number (count n)) (bv := integer target)
      (Evaluates.lookup (by simp [ready, setLocal])) (Evaluates.lookup getTarget') rfl
  have enter : ∀ extra,
      BlockReturns program ready (loopBody.drop 2 ++ [Calls.loopContinuation loopBody statement])
        ((heap ++ primeExtra) ++ countExtra) (number (answer target))
        (((heap ++ primeExtra) ++ countExtra) ++ extra) →
      BlockReturns program env [statement] heap (number (answer target))
        (heap ++ (primeExtra ++ countExtra ++ extra)) := by
    intro extra rest
    apply BlockReturns.while_loop (flag := true) Evaluates.constant rfl
    apply BlockReturns.assign (Evaluates.call (.cons (Evaluates.lookup getN) .nil) primes)
    apply BlockReturns.assign
    · apply Evaluates.call _ counted
      exact .cons (Evaluates.lookup (by simpa [setLocal] using getN)) (.cons (Evaluates.lookup rfl) .nil)
    · simpa [List.append_assoc] using rest
  by_cases over : target < (count n : Int)
  · have same : n = answer target := by
      have least := answer_le target n ⟨lower, over⟩
      omega
    refine ⟨primeExtra ++ countExtra ++ [], enter [] ?_⟩
    apply BlockReturns.branch (flag := true) (by simpa [over] using check) rfl
    simp only [List.append_nil, List.append_assoc]
    exact BlockReturns.return_value (Evaluates.lookup (by simpa only [same] using getN'))
  · have before : n < answer target := by
      have exceeds := (answer_spec target).2
      by_contra notBefore
      have same : n = answer target := by omega
      exact over (by simpa [same] using exceeds)
    obtain ⟨extra, rest⟩ := loop target (n+1) ((heap ++ primeExtra) ++ countExtra)
      (setLocal ready "n" (number (n+1)))
      (by simp [setLocal]) (by simpa [setLocal] using getTarget') (by omega) (by omega)
    refine ⟨primeExtra ++ countExtra ++ extra, enter extra ?_⟩
    apply BlockReturns.branch (flag := false) (by simpa [over] using check) rfl
    simp only [Bool.false_eq_true, if_false, List.nil_append, List.append_assoc]
    apply BlockReturns.augAssign (value := number (n+1)) (op := .add)
      (middle := heap ++ (primeExtra ++ countExtra)) (afterOperation := heap ++ (primeExtra ++ countExtra))
      getN' (Evaluates.constant (value := .int 1))
      (by simp [inplace, Mutable.binary, scalarBinary, number])
    simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, List.append_assoc] using rest
termination_by answer target-n
decreasing_by omega

theorem result (target : Int) (heap : Mutable.Heap) :
    ∃ extra, ReturnsValue program "solve" [integer target] heap (number (answer target)) (heap ++ extra) := by
  let args := ((findFunctionIn program.body "solve").getD ([], [])).1
  let body := ((findFunctionIn program.body "solve").getD ([], [])).2
  let env := functionEnv args body [integer target] []
  obtain ⟨extra, executed⟩ := loop target 2 heap (setLocal env "n" (number 2)) rfl rfl (by omega) (answer_spec target).1
  refine ⟨extra, ?_⟩
  exact BlockReturns.function rfl rfl (BlockReturns.assign Evaluates.constant executed)

end solvers.«77».solve
