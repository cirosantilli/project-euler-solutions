import ProjectEulerSolutions.Lib.TruncatablePrimeRoots
import lang.py.FormalSW.LoopProgress

namespace ProjectEulerSolutions.TruncatablePrimeCalls
open FormalSW.PyAst FormalSW.PyAst.HeapCalls ProjectEulerStatements.TruncatablePrimeBase

set_option Elab.async false

def setup : List Stmt :=
  [.assign [.name "frontier"] rootExpr, .assign [.name "total"] (.constant (.int 0))]

def searchBody (predicateName : String) : List Stmt :=
  [.assign [.name "following"] (.list []),
   .forIn (.name "prefix") (.name "frontier") (parentBody predicateName) [],
   .assign [.name "frontier"] (.name "following")]

def searchLoop (predicateName : String) : Stmt :=
  .whileDo (.name "frontier") (searchBody predicateName) []

def tailCode (predicateName : String) : List Stmt :=
  [searchLoop predicateName, .return (.name "total")]

def totalAt (radix depth : Nat) : Nat := weight radix (candidates radix depth)

theorem totalAt_zero (radix : Nat) : totalAt radix 0 = 0 := by
  simp [totalAt, candidates, weight]

theorem totalAt_step (radix depth : Nat) :
    totalAt radix (depth + 1) = totalAt radix depth + weight radix (level radix (depth + 1)) := by
  simp [totalAt, candidates, List.range_succ, List.flatMap_append, weight_append]

def readyEnv (env : HeapCalls.Env) (address : Nat) : HeapCalls.Env :=
  setLocal (setLocal env "frontier" (.ref address)) "total" (number 0)

theorem initialize_result (program : Module) (radix : Nat) (valid : 2 ≤ radix)
    (primeResult : ∀ n heap, ReturnsValue program "is_prime" [number n] heap
      (flag (decide (Nat.Prime n))) heap)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (getBase : env "base" = some (number radix)) :
    BlockContinues program env setup heap (readyEnv env heap.length)
      (HeapCollection.working heap (numbers (roots radix)) []) := by
  exact (BlockContinues.assign (root_result program radix valid primeResult heap env
    getBase)).append
    (BlockContinues.assign Evaluates.constant)

section Traversal
variable (program : Module) (radix : Nat) (predicateName : String) (valid : 2 ≤ radix)
variable (primeResult : ∀ n heap, ReturnsValue program "is_prime" [number n] heap
  (flag (decide (Nat.Prime n))) heap)
variable (predicateResult : ∀ n, radix ≤ n → RightPrime radix n → ∀ heap,
  ∃ extra, ReturnsValue program predicateName [number n, number radix] heap
    (flag (leftCheck radix n)) (heap ++ extra))
include valid primeResult predicateResult

theorem frontier_step (depth total address : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (found : heap[address]? = some (.list (numbers (level radix depth))))
    (getBase : env "base" = some (number radix))
    (getTotal : env "total" = some (number total))
    (getFrontier : env "frontier" = some (.ref address)) :
    ∃ finalEnv finalHeap, BlockContinues program env (searchBody predicateName) heap finalEnv
      finalHeap ∧
      finalHeap[heap.length]? = some (.list (numbers (level radix (depth + 1)))) ∧
      finalEnv "frontier" = some (.ref heap.length) ∧
      finalEnv "base" = some (number radix) ∧
      finalEnv "total" = some (number (total + weight radix (level radix (depth + 1)))) := by
  let ready := setLocal env "following" (.ref heap.length)
  have initialized : BlockContinues program env [.assign [.name "following"] (.list [])]
      heap ready (HeapCollection.working heap [] []) := by
    exact BlockContinues.assign (Evaluates.list .nil)
  have parents : ∀ p ∈ level radix depth, 0 < p ∧ RightPrime radix p := by
    intro p member
    obtain ⟨lower, _, right⟩ := (mem_level radix depth p valid).mp member
    exact ⟨lt_of_lt_of_le (Nat.pow_pos (by omega : 0 < radix)) lower, right⟩
  obtain ⟨scannedEnv, garbage, scanned, scannedTotal, scannedStable⟩ :=
    parent_loop program radix predicateName valid primeResult predicateResult
      (level radix depth) (level radix depth) 0 address total parents (by simp)
      heap [] [] ready found
      (by simpa [ready, setLocal] using getBase) (by simpa [ready, setLocal] using getTotal)
      (by simp [ready, setLocal])
  have sequence : Mutable.sequence (.ref address) (HeapCollection.working heap [] []) =
      some (.ok (numbers (level radix depth)), HeapCollection.working heap [] []) := by
    have kept := HeapCollection.prefix_found heap [.list []] address _ found
    simp [Mutable.sequence, HeapCollection.working, kept]
  have executed : BlockContinues program ready
      [.forIn (.name "prefix") (.name "frontier") (parentBody predicateName) []]
      (HeapCollection.working heap [] []) scannedEnv
      (HeapCollection.working heap (numbers (level radix (depth + 1))) garbage) := by
    apply BlockContinues.for_sequence (by intro expressions h; cases h)
      (Evaluates.lookup (by simpa [ready, setLocal] using getFrontier)) sequence
    simpa only [List.nil_append, level] using scanned
  have scannedFollowing : scannedEnv "following" = some (.ref heap.length) :=
    (scannedStable "following" (by decide)).trans (by simp [ready, setLocal])
  let finalEnv := setLocal scannedEnv "frontier" (.ref heap.length)
  refine ⟨finalEnv, HeapCollection.working heap (numbers (level radix (depth + 1))) garbage,
    ?_, HeapCollection.found _ _ _, by simp [finalEnv, setLocal], ?_, ?_⟩
  · have assigned : BlockContinues program scannedEnv [.assign [.name "frontier"] (.name
    "following")]
        (HeapCollection.working heap (numbers (level radix (depth + 1))) garbage) finalEnv
        (HeapCollection.working heap (numbers (level radix (depth + 1))) garbage) :=
      BlockContinues.assign (Evaluates.lookup scannedFollowing)
    exact initialized.append (executed.append assigned)
  · simpa [finalEnv, setLocal, ready] using
      (scannedStable "base" (by decide)).trans (by simpa [ready, setLocal] using getBase)
  · simpa only [finalEnv, setLocal, level] using scannedTotal

theorem search_result (success : Exhausts radix) (depth address : Nat)
    (within : depth ≤ Nat.find success) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (found : heap[address]? = some (.list (numbers (level radix depth))))
    (getBase : env "base" = some (number radix))
    (getTotal : env "total" = some (number (totalAt radix depth)))
    (getFrontier : env "frontier" = some (.ref address)) :
    ∃ finalHeap, BlockReturns program env (tailCode predicateName) heap
      (number (answer radix success)) finalHeap := by
  induction measure : Nat.find success - depth using Nat.strong_induction_on
      generalizing depth address heap env with
  | h size ih =>
      by_cases finished : depth = Nat.find success
      · have empty : level radix depth = [] := finished ▸ Nat.find_spec success
        have same : totalAt radix depth = answer radix success := by
          simp only [totalAt, weight, accepted, answer, finished]
        refine ⟨heap, ?_⟩
        apply BlockReturns.while_loop (middle := heap) (flag := false)
          (Evaluates.lookup getFrontier)
          (by simp [Mutable.truth, found, empty, numbers])
        exact BlockReturns.return_value (Evaluates.lookup (same ▸ getTotal))
      · have earlier : depth < Nat.find success := by omega
        have nonempty : level radix depth ≠ [] := Nat.find_min success earlier
        obtain ⟨nextEnv, nextHeap, advanced, nextFound, nextFrontier, nextBase, nextTotal⟩ :=
          frontier_step program radix predicateName valid primeResult predicateResult
            depth (totalAt radix depth) address heap env found getBase getTotal getFrontier
        have totalCorrect : nextEnv "total" = some (number (totalAt radix (depth + 1))) := by
          simpa only [totalAt_step] using nextTotal
        obtain ⟨finalHeap, returned⟩ := ih (Nat.find success - (depth + 1)) (by omega)
          (depth + 1) heap.length (by omega) nextHeap nextEnv nextFound nextBase totalCorrect
            nextFrontier rfl
        refine ⟨finalHeap, ?_⟩
        apply BlockReturns.while_loop (middle := heap) (flag := true)
          (Evaluates.lookup getFrontier)
          (by simp [Mutable.truth, found, numbers, nonempty])
        simpa [tailCode, searchLoop, searchBody, parentBody, digitBody,
          Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using advanced.returns
            returned

theorem function_result (name : String) (args : List Arg) (functionBody : List Stmt)
    (found : findFunctionIn program.body name = some (args, functionBody))
    (arguments : completeArguments args [number radix] = some [number radix])
    (shape : executableBody functionBody = setup ++ tailCode predicateName)
    (baseInput : functionEnv args functionBody [number radix] [] "base" = some (number radix))
    (success : Exhausts radix) (heap : Mutable.Heap) :
    ∃ finalHeap, ReturnsValue program name [number radix] heap (number (answer radix success))
      finalHeap := by
  let entry := functionEnv args functionBody [number radix] []
  let initialized := readyEnv entry heap.length
  let initialHeap := HeapCollection.working heap (numbers (roots radix)) []
  obtain ⟨finalHeap, returned⟩ := search_result program radix predicateName valid primeResult
    predicateResult
    success 0 heap.length (Nat.zero_le _) initialHeap initialized
    (HeapCollection.found _ _ _) (by simpa [initialized, readyEnv, setLocal] using baseInput)
    (by simp [initialized, readyEnv, setLocal, totalAt_zero]) (by simp [initialized, readyEnv,
      setLocal])
  refine ⟨finalHeap, ?_⟩
  apply BlockReturns.function found arguments
  rw [shape]
  exact (initialize_result program radix valid primeResult heap entry baseInput).returns returned

omit valid primeResult predicateResult in
theorem frontier_chunk (depth address : Nat) (heap finalHeap : Mutable.Heap)
    (env finalEnv : HeapCalls.Env) (nonempty : level radix depth ≠ [])
    (getFrontier : env "frontier" = some (.ref address))
    (found : heap[address]? = some (.list (numbers (level radix depth))))
    (continued : BlockContinues program env (searchBody predicateName) heap finalEnv finalHeap) :
    ∃ next, step program 0 ⟨.block (tailCode predicateName) env, [], heap⟩ = .next next ∧
      Reaches program 0 next ⟨.block (tailCode predicateName) finalEnv, [], finalHeap⟩ := by
  refine ⟨_, rfl, ?_⟩
  apply Reaches.next (following := ⟨.value (.ref address),
    [.branch (searchBody predicateName ++
      [Calls.loopContinuation (searchBody predicateName) (searchLoop predicateName)])
      [] [.return (.name "total")] env], heap⟩)
  · simp [step, getFrontier, searchLoop]
  apply Reaches.next (following := ⟨.block (searchBody predicateName ++ tailCode predicateName) env,
    [], heap⟩)
  · simp [step, run, Mutable.truth, found, numbers, nonempty, tailCode, searchLoop,
      searchBody, parentBody, digitBody, Calls.loopContinuation, Calls.loopControl,
      Calls.loopControlStmt]
  · exact continued (tailCode predicateName) 0 []

theorem function_diverges (name : String) (args : List Arg) (functionBody : List Stmt)
    (found : findFunctionIn program.body name = some (args, functionBody))
    (arguments : completeArguments args [number radix] = some [number radix])
    (shape : executableBody functionBody = setup ++ tailCode predicateName)
    (baseInput : functionEnv args functionBody [number radix] [] "base" = some (number radix))
    (absent : ¬ Exhausts radix) : (HeapCalls.scalarFunction program name [.int radix]).Diverges :=
      by
  let entry := functionEnv args functionBody [number radix] []
  let initialized := readyEnv entry 0
  let initialHeap := HeapCollection.working [] (numbers (roots radix)) []
  let boundary : HeapCalls.Env → Mutable.Heap → State := fun env heap =>
    ⟨.block (tailCode predicateName) env, [], heap⟩
  let invariant : State → Prop := fun state => ∃ depth address env heap,
    env "base" = some (number radix) ∧ env "total" = some (number (totalAt radix depth)) ∧
    env "frontier" = some (.ref address) ∧
    heap[address]? = some (.list (numbers (level radix depth))) ∧ state = boundary env heap
  apply diverges_of_chunks program 0 [] (HeapCalls.scalarStep program)
    (fun state next transition => by simp [HeapCalls.scalarStep, transition]) invariant
  · refine ⟨boundary initialized initialHeap, ?_, ?_⟩
    · exact ⟨0, 0, initialized, initialHeap,
        by simpa [initialized, readyEnv, setLocal] using baseInput,
        by simp [initialized, readyEnv, setLocal, totalAt_zero],
        by simp [initialized, readyEnv, setLocal], HeapCollection.found _ _ _, rfl⟩
    · apply Reaches.next (following := ⟨.block (executableBody functionBody) entry, [], []⟩)
      · simp [step, found, arguments, entry]
      · simpa only [shape, boundary] using
          initialize_result program radix valid primeResult [] entry baseInput (tailCode
            predicateName) 0 []
  · intro state current
    obtain ⟨depth, address, env, heap, getBase, getTotal, getFrontier, located, rfl⟩ := current
    have nonempty : level radix depth ≠ [] := fun empty => absent ⟨depth, empty⟩
    obtain ⟨nextEnv, nextHeap, advanced, nextFound, nextFrontier, nextBase, nextTotal⟩ :=
      frontier_step program radix predicateName valid primeResult predicateResult
        depth (totalAt radix depth) address heap env located getBase getTotal getFrontier
    obtain ⟨following, transition, completed⟩ :=
      frontier_chunk program radix predicateName
        depth address heap nextHeap env nextEnv nonempty getFrontier located advanced
    refine ⟨following, boundary nextEnv nextHeap, transition, completed,
      depth + 1, heap.length, nextEnv, nextHeap, nextBase, ?_, nextFrontier, nextFound, rfl⟩
    simpa only [totalAt_step] using nextTotal

end Traversal
end ProjectEulerSolutions.TruncatablePrimeCalls
