import solvers.«14.py.lean.d».Source
import solvers.«14.py.lean.d».CacheWork
import lang.py.FormalSW.HeapRuntimeCostDictionaries
import lang.py.FormalSW.HeapRuntimeCostSequences
import lang.py.FormalSW.HeapRuntimeCostListWrites
import lang.py.FormalSW.HeapRuntimeCostListRepetition
import lang.py.FormalSW.HeapRuntimeCostUnpacking

/-! Runtime of the final cache read, the nonpositive entry case, and the
source initialization of both explicit caches. -/

namespace solvers.«14».runtimeCost
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerStatements.P14

/-- Return the cached length through the actual source branch and subscript.
The dictionary case pays for the current collision chain and its key widths. -/
theorem length_return (limit n : Nat) (env : Mutable.Env) (heap : Mutable.Heap)
    (smallAddress bigAddress : Nat) (small : List Mutable.Value)
    (big : List (Constant × Mutable.Value)) (valid : Cache.Valid limit small big)
    (smallFound : heap[smallAddress]? = some (.list small))
    (bigFound : heap[bigAddress]? = some (.dict big))
    (smallLocal : env "cache_small" = some (.ref smallAddress))
    (bigLocal : env "cache_big" = some (.ref bigAddress))
    (limitLocal : env "limit" = some (.scalar (.int limit)))
    (originLocal : env "n" = some (.scalar (.int n)))
    (known : Cache.Known limit small big n) :
    Runtime.BlockReturnsIn .word program env lengthReturnBody heap
      (50 * (big.length + 1)) (.scalar (.int (collatzLen n))) heap := by
  have test : Runtime.EvaluatesIn .word program env
      (.compare (.name "n") [(.ltE, .name "limit")]) heap 7
      (.scalar (.bool (decide (n ≤ limit)))) heap := by
    apply Runtime.EvaluatesIn.compare (ca := 1) (cb := 1) (work := 2)
      (Runtime.EvaluatesIn.lookup originLocal) (Runtime.EvaluatesIn.lookup limitLocal)
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    · rfl
  by_cases inside : n ≤ limit
  · apply Runtime.BlockReturnsIn.mono (small := 20) _ (by omega)
    apply Runtime.BlockReturnsIn.branch (cc := 7) (cb := 11)
      test (flag := true) (by simp [Mutable.truth, scalarTruth, inside])
    apply Runtime.BlockReturnsIn.return_value (cost := 10)
    apply Runtime.EvaluatesIn.list_subscript (ca := 1) (cb := 1)
      (by intro a b c; simp) (Runtime.EvaluatesIn.lookup smallLocal)
      (Runtime.EvaluatesIn.lookup originLocal) smallFound
    have bound : n < small.length := by rw [valid.size]; omega
    have found := List.getElem?_eq_some_iff.mp
      (show small[n]? = some (.scalar (.int (collatzLen n))) by
        simpa [Cache.Known, Cache.number, inside] using known)
    simpa only [found.2] using Mutable.getItem_list heap smallAddress n small smallFound bound
  · apply Runtime.BlockReturnsIn.mono
      (small := 7 + ((1 + 1 + PrimitiveWork.mapLookup .collisionChain big.length
        (MapWork.keyWork .word big (.int n)) + 5) + 1) + 2)
    · apply Runtime.BlockReturnsIn.branch (cc := 7)
        (cb := (1 + 1 + PrimitiveWork.mapLookup .collisionChain big.length
          (MapWork.keyWork .word big (.int n)) + 5) + 1)
        test (flag := false) (by simp [Mutable.truth, scalarTruth, inside])
      apply Runtime.BlockReturnsIn.return_value
        (cost := 1 + 1 + PrimitiveWork.mapLookup .collisionChain big.length
          (MapWork.keyWork .word big (.int n)) + 5)
      apply Runtime.EvaluatesIn.dictionary_subscript (ca := 1) (cb := 1)
        (by intro a b c; simp) (Runtime.EvaluatesIn.lookup bigLocal)
        (Runtime.EvaluatesIn.lookup originLocal) bigFound
      simpa [Cache.Known, Cache.number, inside] using known
    · have width := valid.keyWork_word_le (n : Int)
      have work := Nat.mul_le_mul_left (big.length + 1)
        (by omega : MapWork.keyWork .word big (.int n) + 1 ≤ 3)
      simp only [PrimitiveWork.mapLookup]
      omega

/-- The early return includes function entry, comparison, branch, and return.
It leaves every existing heap object unchanged. -/
theorem solve.nonpositive_runtime (limit : Int) (nonpositive : limit ≤ 0)
    (heap : Mutable.Heap) :
    Runtime.ReturnsIn .word program "solve" [.scalar (.int limit)] heap 14
      (.scalar (.int 0)) heap := by
  let definition : List Arg × List Stmt := (findFunctionIn program.body "solve").getD ([], [])
  apply Runtime.BlockReturnsIn.function (args := definition.1) (body := definition.2)
    (budget := 11) rfl rfl
  apply Runtime.BlockReturnsIn.branch (cc := 7) (cb := 2)
    (value := .bool true) (flag := true) (middle := heap)
  · apply Runtime.EvaluatesIn.compare (ca := 1) (cb := 1) (work := 2)
      (Runtime.EvaluatesIn.lookup rfl) Runtime.EvaluatesIn.constant
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp, nonpositive]
    · rfl
  · rfl
  · exact Runtime.BlockReturnsIn.return_value (cost := 1) Runtime.EvaluatesIn.constant

/-- After the source has allocated `[0] * (limit + 1)`, execute the remaining
four initialization statements: seed the base case, allocate the empty big
cache, and initialize the winning vertex and length. -/
theorem search_initialize_tail (limit : Nat) (positive : 0 < limit) :
    Runtime.BlockContinuesIn .word program
      (setLocal (functionEnv searchDefinition.1 searchDefinition.2 [.scalar (.int limit)] [])
        "cache_small" (.ref 1))
      (((executableBody searchDefinition.2).take 5).drop 1)
      [.list [.scalar (.int 0)], .list (List.replicate (limit + 1) (.scalar (.int 0)))]
      23 (searchEnv limit) (searchHeap limit) := by
  let entry := functionEnv searchDefinition.1 searchDefinition.2 [.scalar (.int limit)] []
  let allocated := setLocal entry "cache_small" (.ref 1)
  let zeros : List Mutable.Value := List.replicate (limit + 1) (.scalar (.int 0))
  let firstHeap : Mutable.Heap := [.list [.scalar (.int 0)], .list zeros]
  let initializedHeap : Mutable.Heap := [.list [.scalar (.int 0)], .list (Cache.initialSmall limit)]
  have seed : Runtime.BlockContinuesIn .word program allocated
      [.assign [.subscript (.name "cache_small") (.constant (.int 1))] (.constant (.int 1))]
      firstHeap 13 allocated initializedHeap := by
    have written : Runtime.BlockContinuesIn .word program allocated
        [.assign [.subscript (.name "cache_small") (.constant (.int 1))] (.constant (.int 1))]
        firstHeap 13 allocated
        (firstHeap.set 1 (.list (zeros.set 1 (.scalar (.int 1))))) := by
      apply Runtime.BlockContinuesIn.list_setItem (globals := []) (arithmetic := .word)
        (cr := 1) (ca := 5) (address := 1) (offset := 1) (position := 1)
        (items := zeros) (afterRhs := firstHeap) (afterArgs := firstHeap)
        (by intro a b c; simp) Runtime.EvaluatesIn.constant
        (.cons (Runtime.EvaluatesIn.lookup rfl) (.cons Runtime.EvaluatesIn.constant .nil)) rfl
      exact Cache.write_small firstHeap 1 1 zeros (.scalar (.int 1)) rfl
        (by simp [zeros]; omega)
    simpa [firstHeap, initializedHeap, Cache.initialSmall, Cache.number, zeros] using written
  apply Runtime.BlockContinuesIn.append (ca := 13) (cb := 10)
    (body := [.assign [.subscript (.name "cache_small") (.constant (.int 1))]
      (.constant (.int 1))]) (middleEnv := allocated) (middleHeap := initializedHeap) seed
  apply Runtime.BlockContinuesIn.append (ca := 4) (cb := 6)
    (body := [.annAssign (.name "cache_big")
      (.subscript (.name "dict") (.tuple [.name "int", .name "int"])) (some (.dict []))])
    (middleEnv := setLocal allocated "cache_big" (.ref 2)) (middleHeap := searchHeap limit)
  · apply Runtime.BlockContinuesIn.annAssign (cost := 2)
    exact Runtime.EvaluatesIn.dictionary_constants (globals := []) []
  · apply Runtime.BlockContinuesIn.append (ca := 3) (cb := 3)
      (body := [.assign [.name "best_n"] (.constant (.int 1))])
      (middleEnv := setLocal (setLocal allocated "cache_big" (.ref 2))
        "best_n" (.scalar (.int 1))) (middleHeap := searchHeap limit)
    · exact Runtime.BlockContinuesIn.assign (cost := 1) Runtime.EvaluatesIn.constant
    · exact Runtime.BlockContinuesIn.assign (cost := 1) Runtime.EvaluatesIn.constant

/-- Execute all five source initialization statements. Repetition pays for
copying the zero references and for the new list's allocation potential. -/
theorem search_initialize (limit : Nat) (positive : 0 < limit) :
    Runtime.BlockContinuesIn .word program
      (functionEnv searchDefinition.1 searchDefinition.2 [.scalar (.int limit)] [])
      ((executableBody searchDefinition.2).take 5) [] (100 * (limit + 1))
      (searchEnv limit) (searchHeap limit) := by
  let entry := functionEnv searchDefinition.1 searchDefinition.2 [.scalar (.int limit)] []
  let allocated := setLocal entry "cache_small" (.ref 1)
  let zeros : List Mutable.Value := List.replicate (limit + 1) (.scalar (.int 0))
  let firstHeap : Mutable.Heap := [.list [.scalar (.int 0)], .list zeros]
  have singleton : Runtime.EvaluatesIn .word program entry (.list [.constant (.int 0)])
      [] 7 (.ref 0) [.list [.scalar (.int 0)]] := by
    exact Runtime.EvaluatesIn.list (budget := 2) (.cons Runtime.EvaluatesIn.constant .nil)
  have count : Runtime.EvaluatesIn .word program entry
      (.binOp (.name "limit") .add (.constant (.int 1))) [.list [.scalar (.int 0)]]
      6 (.scalar (.int (limit + 1 : Nat))) [.list [.scalar (.int 0)]] := by
    apply Runtime.EvaluatesIn.binary (ca := 1) (cb := 1) (work := 1)
      (Runtime.EvaluatesIn.lookup rfl) Runtime.EvaluatesIn.constant
    · simp [Mutable.binary, scalarBinary]
    · rfl
  have allocatedList : Runtime.EvaluatesIn .word program entry
      (.binOp (.list [.constant (.int 0)]) .mult
        (.binOp (.name "limit") .add (.constant (.int 1))))
      [] (2 * (limit + 1) + 18) (.ref 1) firstHeap := by
    have repeated := Runtime.EvaluatesIn.list_repeat (ca := 7) (cb := 6)
      (items := [.scalar (.int 0)]) singleton count rfl
    simpa [firstHeap, zeros] using repeated.mono
      (large := 2 * (limit + 1) + 18) (by simp [PrimitiveWork.integer]; omega)
  apply Runtime.BlockContinuesIn.mono
    (small := (2 * (limit + 1) + 18 + 2) + 23) _ (by omega)
  apply Runtime.BlockContinuesIn.append (ca := 2 * (limit + 1) + 18 + 2) (cb := 23)
    (body := [.assign [.name "cache_small"]
      (.binOp (.list [.constant (.int 0)]) .mult
        (.binOp (.name "limit") .add (.constant (.int 1))))])
    (middleEnv := allocated) (middleHeap := firstHeap)
  · exact Runtime.BlockContinuesIn.assign (cost := 2 * (limit + 1) + 18) allocatedList
  · exact search_initialize_tail limit positive

end solvers.«14».runtimeCost
