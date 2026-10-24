import ProjectEulerSolutions.Lib.TruncatablePrimeCalls

namespace ProjectEulerSolutions.TruncatablePrimeCalls
open FormalSW.PyAst FormalSW.PyAst.HeapCalls ProjectEulerStatements.TruncatablePrimeBase

set_option Elab.async false

def rootBody (address : Nat) : List Stmt :=
  comprehensionFilters [.call (.name "is_prime") [.name "n"]]
    [.comprehensionAppend address (.name "n")]

private theorem root_loop (program : Module)
    (primeResult : ∀ n heap, ReturnsValue program "is_prime" [number n] heap
      (flag (decide (Nat.Prime n))) heap)
    (first count stop : Nat) (endpoint : first + count = stop)
    (base : Mutable.Heap) (before : List Nat) (env : HeapCalls.Env) :
    ∃ finalEnv, BlockContinues program env
      [.rangeStepLoop "n" first stop 1 (rootBody base.length) []]
      (HeapCollection.working base (numbers before) []) finalEnv
      (HeapCollection.working base
        (numbers (before ++ (List.range' first count).filter fun n => decide (Nat.Prime n))) []) :=
          by
  induction count generalizing first before env with
  | zero =>
      refine ⟨env, ?_⟩
      simpa only [List.range'_zero, List.filter_nil, List.append_nil] using
        (BlockContinues.range_done (module := program) (env := env)
          (heap := HeapCollection.working base (numbers before) [])
          (name := "n") (next := first) (stop := stop) (stride := 1)
          (body := rootBody base.length) (otherwise := [])
          (by simp [Calls.rangeContinues]; omega) .nil)
  | succ count ih =>
      let bound := setLocal env "n" (number first)
      let heap := HeapCollection.working base (numbers before) []
      have tested : Evaluates program bound (.call (.name "is_prime") [.name "n"]) heap
          (flag (decide (Nat.Prime first))) heap :=
        Evaluates.call (.cons (Evaluates.lookup (by simp [bound, setLocal])) .nil)
          (primeResult first heap)
      by_cases prime : Nat.Prime first
      · have emitted : BlockContinues program bound (rootBody base.length) heap bound
            (HeapCollection.working base (numbers (before ++ [first])) []) := by
          apply BlockContinues.branch (middle := heap) (flag := true) (value := flag true)
            (by simpa only [prime, decide_true] using tested) rfl
          have appended := BlockContinues.comprehension_append (globals := [])
            (Evaluates.lookup (module := program) (env := bound) (heap := heap)
              (name := "n") (value := number first) (by simp [bound, setLocal]))
            (HeapCollection.found base (numbers before) [])
          simpa only [heap, HeapCollection.store, numbers, List.map_append, List.map_singleton]
            using appended
        obtain ⟨finalEnv, rest⟩ := ih (first + 1) (by omega) (before ++ [first]) bound
        refine ⟨finalEnv, ?_⟩
        apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        simpa [rootBody, comprehensionFilters, Calls.loopContinuation, Calls.loopControl,
          Calls.loopControlStmt, List.range'_succ, List.append_assoc, prime] using emitted.append
            rest
      · obtain ⟨finalEnv, rest⟩ := ih (first + 1) (by omega) before bound
        refine ⟨finalEnv, ?_⟩
        apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        have skipped : BlockContinues program bound (rootBody base.length) heap bound heap := by
          apply BlockContinues.branch (middle := heap) (flag := false) (value := flag false)
            (by simpa only [prime, decide_false] using tested) rfl
          exact .nil
        simpa [rootBody, comprehensionFilters, Calls.loopContinuation, Calls.loopControl,
          Calls.loopControlStmt, List.range'_succ, prime] using skipped.append rest

theorem roots_eq_range (radix : Nat) (valid : 2 ≤ radix) :
    roots radix = (List.range' 2 (radix - 2)).filter fun n => decide (Nat.Prime n) := by
  apply (List.pairwise_lt_range.filter _).eq_of_mem_iff
    ((List.pairwise_lt_range' 1 (by decide)).filter _)
  intro n
  simp only [List.mem_filter, List.mem_range, List.mem_range'_1, decide_eq_true_eq]
  constructor
  · rintro ⟨inside, prime⟩
    exact ⟨⟨prime.two_le, by omega⟩, prime⟩
  · rintro ⟨⟨_, inside⟩, prime⟩
    exact ⟨by omega, prime⟩

def rootExpr : Expr := .listComp (.name "n")
  [("n", .call (.name "range") [.constant (.int 2), .name "base"],
    [.call (.name "is_prime") [.name "n"]])]

theorem root_result (program : Module) (radix : Nat) (valid : 2 ≤ radix)
    (primeResult : ∀ n heap, ReturnsValue program "is_prime" [number n] heap
      (flag (decide (Nat.Prime n))) heap)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (getBase : env "base" = some (number radix)) :
    Evaluates program env rootExpr heap (.ref heap.length)
      (HeapCollection.working heap (numbers (roots radix)) []) := by
  let generators : List (String × Expr × List Expr) :=
    [("n", .call (.name "range") [.constant (.int 2), .name "base"],
      [.call (.name "is_prime") [.name "n"]])]
  obtain ⟨finalEnv, collected⟩ := root_loop program primeResult 2 (radix - 2) radix (by omega)
    heap [] (comprehensionEnv env generators)
  apply Evaluates.list_comp_range (values := [number 2, number radix])
    (middle := heap) (finalEnv := finalEnv)
    (.cons Evaluates.constant (.cons (Evaluates.lookup getBase) .nil)) rfl
  simpa only [rootBody, generators, rootExpr, List.nil_append,
    roots_eq_range radix valid] using collected

end ProjectEulerSolutions.TruncatablePrimeCalls
