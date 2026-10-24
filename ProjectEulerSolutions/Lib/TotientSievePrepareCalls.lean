import ProjectEulerSolutions.Lib.TotientSieveLoopCalls
import ProjectEulerStatements.Lib.TotientLinearCache

/-! Primality detection, cache append, and prime-cell initialization in the source sieve. -/
namespace ProjectEulerSolutions.TotientSieveCalls
open FormalSW.PyAst HeapCalls ProjectEulerStatements.TotientLinearSieve
set_option Elab.async false
variable {program : Module}

abbrev primeCondition : Expr := .unaryOp .not (.subscript (.name "is_comp") (.name "i"))
abbrev cacheAppend : Stmt :=
  .expr (.call (.attribute (.name "primes") "append") [.name "i"])
abbrev initializeExpression : Expr := .binOp (.name "i") .sub (.constant (.int 1))
abbrev initializeCode : Stmt :=
  .assign [.subscript (.name "phi") (.name "i")] initializeExpression

abbrev outerBody : List Stmt :=
  [.ifThen primeCondition [cacheAppend, initializeCode] [], innerLoop]

theorem outer_source : outerBody =
    [.ifThen primeCondition [cacheAppend, initializeCode] [], innerLoop] := rfl

theorem prime_condition_result (limit i : Nat) (table : Table) (base : Mutable.Heap)
    (cache : List Nat) (env : HeapCalls.Env) (within : i ≤ limit)
    (ready : Ready limit i base.length env) :
    Evaluates program env primeCondition (realize base (limit + 1) table cache)
      (.scalar (.bool (!(table i).2))) (realize base (limit + 1) table cache) := by
  apply Evaluates.unary (value := markValue ((table i).2))
    (middle := realize base (limit + 1) table cache)
  · apply Evaluates.subscript (by intro a b c impossible; cases impossible)
      (Evaluates.lookup ready.getMarks) (Evaluates.lookup ready.getI)
    simpa [entries] using Mutable.getItem_list (realize base (limit + 1) table cache)
      (base.length + 3) i (entries (limit + 1) (fun k => markValue ((table k).2)))
      (marks_found base (limit + 1) _ _ cache) (by simpa [entries] using within)
  · cases (table i).2 <;> simp [HeapCalls.unary, Mutable.truth, markValue, scalarTruth]

theorem prepare_heap (base : Mutable.Heap) (size i : Nat) (table : Table) (cache : List Nat)
    (flag : (table i).2 = false) :
    working base size (Function.update (fun n => (table n).1) i (i - 1))
      (fun n => (table n).2) cache = realize base size (prepare i table) cache := by
  have first : Function.update (fun n => (table n).1) i (i - 1) =
      fun n => (prepare i table n).1 := by
    funext n
    by_cases same : n = i <;> simp [prepare, flag, Function.update_apply, same]
  have second : (fun n => (table n).2) = fun n => (prepare i table n).2 := by
    funext n
    by_cases same : n = i <;> simp [prepare, flag, Function.update_apply, same]
  rw [first, second]

theorem initialize_result (i : Nat) (large : 2 ≤ i) (heap : Mutable.Heap)
    (env : HeapCalls.Env) (getI : env "i" = some (number i)) :
    Evaluates program env initializeExpression heap (number (i - 1)) heap := by
  exact Evaluates.binary (Evaluates.lookup getI) Evaluates.constant
    (by simp [Mutable.binary, scalarBinary, number, Nat.cast_sub (by omega : 1 ≤ i)])

/-- The live composite mark determines exactly the prime-cache and totient initialization. -/
theorem prepare_result (limit i : Nat) (table : Table) (cache : List Nat)
    (large : 2 ≤ i) (within : i ≤ limit) (current : table i = expected i i)
    (base : Mutable.Heap) (env : HeapCalls.Env) (ready : Ready limit i base.length env) :
    BlockContinues program env [.ifThen primeCondition [cacheAppend, initializeCode] []]
      (realize base (limit + 1) table cache) env
      (realize base (limit + 1) (prepare i table) (nextCache i cache)) := by
  have flag := congrArg Prod.snd current
  rw [expected_mark i large] at flag
  have tested := prime_condition_result (program := program)
    limit i table base cache env within ready
  by_cases prime : Nat.Prime i
  · have falseFlag : (table i).2 = false := by
      simpa only [prime, decide_true, Bool.not_true] using flag
    have append : Evaluates program env
        (.call (.attribute (.name "primes") "append") [.name "i"])
        (realize base (limit + 1) table cache) (.scalar .none)
        (realize base (limit + 1) table (cache ++ [i])) := by
      have found := cache_found base (limit + 1) (fun n => (table n).1)
        (fun n => (table n).2) cache
      have executed := Evaluates.append (module := program) (globals := []) (env := env)
        (Evaluates.lookup ready.getCache) found (Evaluates.lookup ready.getI) found
      have encoded : cache.map number ++ [number i] = (cache ++ [i]).map number := by simp
      simpa only [encoded, store_cache] using executed
    have initialized := set_phi (program := program)
      base (limit + 1) i (i - 1) (fun n => (table n).1)
      (fun n => (table n).2) (cache ++ [i]) env "i" initializeExpression (by omega)
      ready.getPhi ready.getI (initialize_result (program := program) i large _ env ready.getI)
    rw [prepare_heap base (limit + 1) i table (cache ++ [i]) falseFlag] at initialized
    simp only [nextCache, if_pos prime]
    apply BlockContinues.branch_next (flag := true)
      (by simpa only [falseFlag, Bool.not_false] using tested) rfl
    exact (BlockContinues.discard append).append initialized
  · have trueFlag : (table i).2 = true := by
      simpa only [prime, decide_false, Bool.not_false] using flag
    simp only [nextCache, if_neg prime, prepare, trueFlag, if_true]
    apply BlockContinues.branch_next (flag := false)
      (by simpa only [trueFlag, Bool.not_true] using tested) rfl
    exact .nil

end ProjectEulerSolutions.TotientSieveCalls
