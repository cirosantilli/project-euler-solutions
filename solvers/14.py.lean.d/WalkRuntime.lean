import solvers.«14.py.lean.d».Source
import solvers.«14.py.lean.d».CacheWork
import lang.py.FormalSW.HeapRuntimeCostDictionaries
import lang.py.FormalSW.HeapRuntimeCostLoopControl
import lang.py.FormalSW.HeapRuntimeCostRanges

/-! Runtime of the actual forward cache walk. Arithmetic uses word tariffs;
dictionary probes pay for the live collision chain, and path appends retain
the dynamic-array potential charged by `EvaluatesIn.append`. -/

namespace solvers.«14».runtimeCost
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerStatements.P14 ProjectEulerStatements.Collatz

/-- The source bit test, both arithmetic branches, and the assignment. -/
theorem advance_continues (n : Nat) (env : Mutable.Env) (heap : Mutable.Heap)
    (current : env "m" = some (.scalar (.int n))) :
    Runtime.BlockContinuesIn .word program env advanceBody heap 30
      (setLocal env "m" (.scalar (.int (collatzStep n)))) heap := by
  have test : Runtime.EvaluatesIn .word program env
      (.binOp (.name "m") .bitAnd (.constant (.int 1))) heap 6
      (.scalar (.int (↑(n % 2)))) heap := by
    apply Runtime.EvaluatesIn.binary (ca := 1) (cb := 1) (work := 1)
      (Runtime.EvaluatesIn.lookup current) Runtime.EvaluatesIn.constant
    · simp [Mutable.binary, scalarBitAnd]
    · rfl
  by_cases even : n % 2 = 0
  · apply Runtime.BlockContinuesIn.mono (small := 15) _ (by omega)
    apply Runtime.BlockContinuesIn.branch (cc := 6) (cb := 7) test (flag := false)
      (by simp [Mutable.truth, scalarTruth, even])
    simp only [Bool.false_eq_true, ↓reduceIte, collatzStep, even, ↓reduceIte]
    apply Runtime.BlockContinuesIn.augAssign (cost := 1) (work := 1)
      current Runtime.EvaluatesIn.constant
    · simp [inplace, Mutable.binary, scalarBinary]
    · rfl
  · have odd : n % 2 = 1 := by omega
    apply Runtime.BlockContinuesIn.mono (small := 21) _ (by omega)
    apply Runtime.BlockContinuesIn.branch (cc := 6) (cb := 13) test (flag := true)
      (by simp [Mutable.truth, scalarTruth, odd])
    simp only [↓reduceIte, collatzStep, even, ↓reduceIte]
    apply Runtime.BlockContinuesIn.assign (cost := 11)
    apply Runtime.EvaluatesIn.binary (ca := 6) (cb := 1) (work := 1)
      (av := .int (3 * n)) (bv := .int 1) (middle := heap) (afterRight := heap)
    · apply Runtime.EvaluatesIn.binary (ca := 1) (cb := 1) (work := 1)
        Runtime.EvaluatesIn.constant (Runtime.EvaluatesIn.lookup current)
      · simp [Mutable.binary, scalarBinary]
      · rfl
    · exact Runtime.EvaluatesIn.constant
    · simp [Mutable.binary, scalarBinary]
    · rfl

private theorem bound_test (limit n : Nat) (env : Mutable.Env) (heap : Mutable.Heap)
    (limitLocal : env "limit" = some (.scalar (.int limit)))
    (vertexLocal : env "m" = some (.scalar (.int n))) :
    Runtime.EvaluatesIn .word program env
      (.compare (.name "m") [(.ltE, .name "limit")]) heap 7
      (.scalar (.bool (decide (n ≤ limit)))) heap := by
  apply Runtime.EvaluatesIn.compare (ca := 1) (cb := 1) (work := 2)
    (Runtime.EvaluatesIn.lookup vertexLocal) (Runtime.EvaluatesIn.lookup limitLocal)
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  · rfl

/-- The live cache invariant bounds key widths independently of the table size. -/
theorem lookup_work_le (limit n : Nat) (small : List Mutable.Value)
    (big : List (Constant × Mutable.Value)) (valid : Cache.Valid limit small big) :
    PrimitiveWork.mapLookup .collisionChain big.length (MapWork.keyWork .word big (.int n)) ≤
      3 * (big.length + 1) := by
  have width := valid.keyWork_word_le (n : Int)
  calc
    PrimitiveWork.mapLookup .collisionChain big.length (MapWork.keyWork .word big (.int n))
        ≤ (big.length + 1) * 3 :=
      Nat.mul_le_mul_left _ (by omega : MapWork.keyWork .word big (.int n) + 1 ≤ 3)
    _ = 3 * (big.length + 1) := Nat.mul_comm _ _

/-- A small-cache read pays for index normalization as well as the list access. -/
theorem small_read (n : Nat) (env : Mutable.Env) (heap : Mutable.Heap)
    (address : Nat) (small : List Mutable.Value) (value : Mutable.Value)
    (found : heap[address]? = some (.list small))
    (lookupLocal : env "cache_small" = some (.ref address))
    (current : env "m" = some (.scalar (.int n)))
    (present : small[n]? = some value) :
    Runtime.EvaluatesIn .word program env (.subscript (.name "cache_small") (.name "m"))
      heap 10 value heap := by
  apply Runtime.EvaluatesIn.list_subscript (ca := 1) (cb := 1)
    (by intro a b c; simp) (Runtime.EvaluatesIn.lookup lookupLocal)
    (Runtime.EvaluatesIn.lookup current) found
  have entry := List.getElem?_eq_some_iff.mp present
  simpa only [entry.2] using Mutable.getItem_list heap address n small found entry.1

/-- Both dictionary hits and misses use the same size-sensitive lookup tariff. -/
theorem big_read (limit n : Nat) (env : Mutable.Env) (heap : Mutable.Heap)
    (address : Nat) (small : List Mutable.Value) (big : List (Constant × Mutable.Value))
    (valid : Cache.Valid limit small big) (value : Mutable.Value)
    (found : heap[address]? = some (.dict big))
    (lookupLocal : env "cache_big" = some (.ref address))
    (current : env "m" = some (.scalar (.int n)))
    (resultValue : (ScalarDict.lookup big (.int n)).getD (.scalar .none) = value) :
    Runtime.EvaluatesIn .word program env
      (.call (.attribute (.name "cache_big") "get") [.name "m"])
      heap (3 * (big.length + 1) + 6) value heap := by
  apply Runtime.EvaluatesIn.mono
    (small := 1 + 2 + PrimitiveWork.mapLookup .collisionChain big.length
      (MapWork.keyWork .word big (.int n)) + 3)
  · apply Runtime.EvaluatesIn.dictionary_get (cr := 1) (ca := 2) (key := .int n)
      (by intro name same; cases same; simp [lookupLocal])
      (Runtime.EvaluatesIn.lookup lookupLocal) found
      (.cons (Runtime.EvaluatesIn.lookup current) .nil) found (Or.inl rfl)
    simp [primitive, Mutable.dictionaryGet, found, resultValue]
  · have work := lookup_work_le limit n small big valid
    omega

/-- Execute the generated `cm is not None` expression, including both argument
frames and the unary operation. The premise is only the scalar identity result. -/
private theorem not_none (env : Mutable.Env) (heap : Mutable.Heap)
    (value : Constant) (same : Bool) (lookupLocal : env "cm" = some (.scalar value))
    (identityResult : identity (.scalar value) (.scalar .none) heap =
      some (.ok (.scalar (.bool same)), heap)) :
    Runtime.EvaluatesIn .word program env
      (.unaryOp .not (.builtinCall .identity [.name "cm", .constant .none]))
      heap 10 (.scalar (.bool (!same))) heap := by
  intro anchors stack
  have enter : Runtime.Reaches (globals := []) .word program anchors
      ⟨.expr (.unaryOp .not (.builtinCall .identity [.name "cm", .constant .none])) env,
        stack, heap⟩ 1
      ⟨.expr (.builtinCall .identity [.name "cm", .constant .none]) env,
        .unary .not :: stack, heap⟩ := Runtime.advance rfl rfl rfl
  have call : Runtime.Reaches (globals := []) .word program anchors
      ⟨.expr (.builtinCall .identity [.name "cm", .constant .none]) env,
        .unary .not :: stack, heap⟩ 1
      (argumentsState (.builtin .identity) env [] [.name "cm", .constant .none]
        (.unary .not :: stack) heap) := Runtime.advance rfl rfl rfl
  have args : Runtime.ArgumentsIn (globals := []) .word program env 0
      [.name "cm", .constant .none] heap 5 [.scalar value, .scalar .none] heap :=
    .cons (Runtime.EvaluatesIn.lookup lookupLocal) (.cons Runtime.EvaluatesIn.constant .nil)
  have invoke : Runtime.Reaches (globals := []) .word program anchors
      ⟨.operation (.builtin .identity) [.scalar value, .scalar .none],
        .unary .not :: stack, heap⟩ 2
      ⟨.value (.scalar (.bool same)), .unary .not :: stack, heap⟩ := by
    apply Runtime.advance
    · simp [step, primitive, result, run, identityResult]
    · cases value <;> rfl
    · rfl
  have negate : Runtime.Reaches (globals := []) .word program anchors
      ⟨.value (.scalar (.bool same)), .unary .not :: stack, heap⟩ 1
      ⟨.value (.scalar (.bool (!same))), stack, heap⟩ := by
    apply Runtime.advance
    · cases same <;> rfl
    · rfl
    · rfl
  exact ((((enter.trans call).trans
    (args.reaches (completed := []) (.builtin .identity) anchors (.unary .not :: stack))).trans
      invoke).trans negate)

/-- A miss completes the source probe without changing any heap object. -/
theorem probe_miss (limit n : Nat) (env : Mutable.Env) (heap : Mutable.Heap)
    (smallAddress bigAddress : Nat) (small : List Mutable.Value)
    (big : List (Constant × Mutable.Value)) (valid : Cache.Valid limit small big)
    (smallFound : heap[smallAddress]? = some (.list small))
    (bigFound : heap[bigAddress]? = some (.dict big))
    (smallLocal : env "cache_small" = some (.ref smallAddress))
    (bigLocal : env "cache_big" = some (.ref bigAddress))
    (limitLocal : env "limit" = some (.scalar (.int limit)))
    (vertexLocal : env "m" = some (.scalar (.int n)))
    (miss : Cache.Missing limit small big n) :
    Runtime.BlockContinuesIn .word program env probeBody heap (50 * (big.length + 1))
      (setLocal env "cm" (missValue limit n)) heap := by
  have test := bound_test limit n env heap limitLocal vertexLocal
  by_cases inside : n ≤ limit
  · apply Runtime.BlockContinuesIn.mono (small := 24) _ (by omega)
    apply Runtime.BlockContinuesIn.branch (cc := 7) (cb := 15) test (flag := true)
      (by simp [Mutable.truth, scalarTruth, inside])
    apply Runtime.BlockContinuesIn.append (ca := 12) (cb := 3)
      (body := [.assign [.name "cm"] (.subscript (.name "cache_small") (.name "m"))])
    · apply Runtime.BlockContinuesIn.assign (cost := 10)
      exact small_read n env heap smallAddress small (.scalar (.int 0)) smallFound
        smallLocal vertexLocal (by simpa [Cache.Missing, Cache.number, inside] using miss)
    · apply Runtime.BlockContinuesIn.branch (cc := 1) (cb := 0)
        (Runtime.EvaluatesIn.lookup rfl) (flag := false) rfl
      simpa [missValue, inside] using
        (Runtime.BlockContinuesIn.nil (globals := []) (arithmetic := .word)
          (module := program) (env := setLocal env "cm" (.scalar (.int 0))) (heap := heap))
  · apply Runtime.BlockContinuesIn.mono
      (small := 7 + ((3 * (big.length + 1) + 6 + 2) + 12) + 2) _ (by omega)
    apply Runtime.BlockContinuesIn.branch (cc := 7) (cb := (3 * (big.length + 1) + 6 + 2) + 12)
      test (flag := false) (by simp [Mutable.truth, scalarTruth, inside])
    apply Runtime.BlockContinuesIn.append (ca := 3 * (big.length + 1) + 6 + 2) (cb := 12)
      (body := [.assign [.name "cm"]
        (.call (.attribute (.name "cache_big") "get") [.name "m"])])
    · apply Runtime.BlockContinuesIn.assign (cost := 3 * (big.length + 1) + 6)
      apply big_read limit n env heap bigAddress small big valid (.scalar .none)
        bigFound bigLocal vertexLocal
      simp [show ScalarDict.lookup big (.int n) = none by
        simpa [Cache.Missing, inside] using miss]
    · apply Runtime.BlockContinuesIn.branch (cc := 10) (cb := 0)
        (not_none _ heap .none true rfl rfl) (flag := false) rfl
      simpa [missValue, inside] using
        (Runtime.BlockContinuesIn.nil (globals := []) (arithmetic := .word)
          (module := program) (env := setLocal env "cm" (.scalar .none)) (heap := heap))

/-- A miss appends one vertex using the existing amortized append rule, then
advances the actual source arithmetic. No cache entry is written. -/
theorem walk_miss (limit n : Nat) (env : Mutable.Env) (heap : Mutable.Heap)
    (smallAddress bigAddress pathAddress : Nat) (small path : List Mutable.Value)
    (big : List (Constant × Mutable.Value)) (valid : Cache.Valid limit small big)
    (smallFound : heap[smallAddress]? = some (.list small))
    (bigFound : heap[bigAddress]? = some (.dict big))
    (pathFound : heap[pathAddress]? = some (.list path))
    (smallLocal : env "cache_small" = some (.ref smallAddress))
    (bigLocal : env "cache_big" = some (.ref bigAddress))
    (pathLocal : env "path" = some (.ref pathAddress))
    (limitLocal : env "limit" = some (.scalar (.int limit)))
    (vertexLocal : env "m" = some (.scalar (.int n)))
    (miss : Cache.Missing limit small big n) :
    Runtime.BlockContinuesIn .word program env walkBody heap (100 * (big.length + 1))
      (setLocal (setLocal env "cm" (missValue limit n)) "m" (.scalar (.int (collatzStep n))))
      (heap.set pathAddress (.list (path ++ [.scalar (.int n)]))) := by
  apply Runtime.BlockContinuesIn.mono (small := 50 * (big.length + 1) + 41) _ (by omega)
  apply Runtime.BlockContinuesIn.append (ca := 50 * (big.length + 1)) (cb := 41)
    (body := probeBody)
    (probe_miss limit n env heap smallAddress bigAddress small big valid smallFound bigFound
      smallLocal bigLocal limitLocal vertexLocal miss)
  apply Runtime.BlockContinuesIn.append (ca := 11) (cb := 30)
    (body := [.expr (.call (.attribute (.name "path") "append") [.name "m"])])
  · apply Runtime.BlockContinuesIn.discard (cost := 9)
    exact Runtime.EvaluatesIn.append (cr := 1) (ci := 1)
      (Runtime.EvaluatesIn.lookup (by simpa [setLocal] using pathLocal)) pathFound
      (Runtime.EvaluatesIn.lookup (by simpa [setLocal] using vertexLocal)) pathFound
  · exact advance_continues n _ _ (by simpa [setLocal] using vertexLocal)

/-- A hit executes the source `break`, skipping the append and arithmetic and
crossing exactly this loop's resume boundary. -/
theorem walk_hit (limit n : Nat) (positive : 0 < n) (env : Mutable.Env) (heap : Mutable.Heap)
    (smallAddress bigAddress : Nat) (small : List Mutable.Value)
    (big : List (Constant × Mutable.Value)) (valid : Cache.Valid limit small big)
    (smallFound : heap[smallAddress]? = some (.list small))
    (bigFound : heap[bigAddress]? = some (.dict big))
    (smallLocal : env "cache_small" = some (.ref smallAddress))
    (bigLocal : env "cache_big" = some (.ref bigAddress))
    (limitLocal : env "limit" = some (.scalar (.int limit)))
    (vertexLocal : env "m" = some (.scalar (.int n)))
    (known : Cache.Known limit small big n) :
    Runtime.BlockContinuesIn .word program env [walkLoop] heap (50 * (big.length + 1))
      (setLocal (setLocal env "cm" (.scalar (.int (collatzLen n))))
        "length" (.scalar (.int (collatzLen n)))) heap := by
  have test := bound_test limit n env heap limitLocal vertexLocal
  by_cases inside : n ≤ limit
  · apply Runtime.BlockContinuesIn.mono (small := 31) _ (by omega)
    apply Runtime.BlockContinuesIn.while_loop (cc := 1) (cb := 28)
      Runtime.EvaluatesIn.constant (flag := true) rfl
    apply Runtime.BlockContinuesIn.branch_tail (cc := 7) (cb := 19)
      test (flag := true) (by simp [Mutable.truth, scalarTruth, inside])
    apply Runtime.BlockContinuesIn.append (ca := 12) (cb := 7)
      (body := [.assign [.name "cm"] (.subscript (.name "cache_small") (.name "m"))])
    · apply Runtime.BlockContinuesIn.assign (cost := 10)
      exact small_read n env heap smallAddress small (.scalar (.int (collatzLen n)))
        smallFound smallLocal vertexLocal
        (by simpa [Cache.Known, Cache.number, inside] using known)
    · apply Runtime.BlockContinuesIn.branch_tail (cc := 1) (cb := 4)
        (Runtime.EvaluatesIn.lookup rfl) (flag := true)
        (by have nonzero := length_positive n positive
            simp [Mutable.truth, scalarTruth]; omega)
      apply Runtime.BlockContinuesIn.append (ca := 3) (cb := 1)
        (body := [.assign [.name "length"] (.name "cm")])
      · exact Runtime.BlockContinuesIn.assign (cost := 1) (Runtime.EvaluatesIn.lookup rfl)
      · intro following anchors stack
        exact Runtime.advance (globals := []) rfl rfl rfl
  · apply Runtime.BlockContinuesIn.mono
      (small := 1 + (7 + ((3 * (big.length + 1) + 6 + 2) + 16) + 2) + 2) _ (by omega)
    apply Runtime.BlockContinuesIn.while_loop (cc := 1)
      (cb := 7 + ((3 * (big.length + 1) + 6 + 2) + 16) + 2)
      Runtime.EvaluatesIn.constant (flag := true) rfl
    apply Runtime.BlockContinuesIn.branch_tail (cc := 7)
      (cb := (3 * (big.length + 1) + 6 + 2) + 16)
      test (flag := false) (by simp [Mutable.truth, scalarTruth, inside])
    apply Runtime.BlockContinuesIn.append (ca := 3 * (big.length + 1) + 6 + 2) (cb := 16)
      (body := [.assign [.name "cm"]
        (.call (.attribute (.name "cache_big") "get") [.name "m"])])
    · apply Runtime.BlockContinuesIn.assign (cost := 3 * (big.length + 1) + 6)
      apply big_read limit n env heap bigAddress small big valid (.scalar (.int (collatzLen n)))
        bigFound bigLocal vertexLocal
      simp [show ScalarDict.lookup big (.int n) = some (.scalar (.int (collatzLen n))) by
        simpa [Cache.Known, Cache.number, inside] using known]
    · apply Runtime.BlockContinuesIn.branch_tail (cc := 10) (cb := 4)
        (not_none _ heap (.int (collatzLen n)) false rfl rfl) (flag := true) rfl
      apply Runtime.BlockContinuesIn.append (ca := 3) (cb := 1)
        (body := [.assign [.name "length"] (.name "cm")])
      · exact Runtime.BlockContinuesIn.assign (cost := 1) (Runtime.EvaluatesIn.lookup rfl)
      · intro following anchors stack
        exact Runtime.advance (globals := []) rfl rfl rfl

/-- The complete forward loop retains the exact functional output, including
the trail and heap layout. Since this loop never writes a cache, the dictionary
size remains fixed throughout the induction. -/
theorem walk_terminates (origin limit smallAddress bigAddress : Nat) (base : Mutable.Heap)
    (small : List Mutable.Value) (big : List (Constant × Mutable.Value))
    (valid : Cache.Valid limit small big)
    (smallFound : base[smallAddress]? = some (.list small))
    (bigFound : base[bigAddress]? = some (.dict big))
    (current : Nat) (path : List Nat) (env : Mutable.Env) (positive : 0 < current)
    (locals : WalkLocals env origin limit smallAddress bigAddress base.length current)
    (trail : Trail origin current path) :
    ∃ stop visited finalEnv,
      Trail origin stop visited ∧ 0 < stop ∧
      WalkLocals finalEnv origin limit smallAddress bigAddress base.length stop ∧
      finalEnv "length" = some (.scalar (.int (collatzLen stop))) ∧
      Cache.Known limit small big stop ∧
      Runtime.BlockContinuesIn .word program env [walkLoop] (base ++ [.list (vertices path)])
        (500 * (big.length + 1) * (collatzLen current + 1))
        finalEnv (base ++ [.list (vertices visited)]) := by
  induction h : collatzMeasure current using Nat.strong_induction_on generalizing current path env with
  | h measure ih =>
      have smallBound := (List.getElem?_eq_some_iff.mp smallFound).1
      have bigBound := (List.getElem?_eq_some_iff.mp bigFound).1
      have smallHere : (base ++ [.list (vertices path)])[smallAddress]? = some (.list small) := by
        simpa [List.getElem?_append_left smallBound] using smallFound
      have bigHere : (base ++ [.list (vertices path)])[bigAddress]? = some (.dict big) := by
        simpa [List.getElem?_append_left bigBound] using bigFound
      rcases valid.classify current with known | missing
      · refine ⟨current, path, _, trail, positive, locals.hit, ?_, known, ?_⟩
        · simp [setLocal]
        · apply Runtime.BlockContinuesIn.mono
            (small := 50 * (big.length + 1))
            (walk_hit limit current positive env _ smallAddress bigAddress small big valid
              smallHere bigHere locals.small locals.big locals.bound locals.current known)
          have enough : 500 * (big.length + 1) ≤
              500 * (big.length + 1) * (collatzLen current + 1) :=
            Nat.le_mul_of_pos_right _ (by omega)
          omega
      · have notOne : current ≠ 1 := valid.miss_not_one current missing
        have decreased : collatzMeasure (collatzStep current) < measure := by
          rw [← h]
          exact collatzMeasure_decreases positive notOne
        obtain ⟨stop, visited, finalEnv, trace, stopPositive, finalLocals, lengthLocal, hit, recurse⟩ :=
          ih (collatzMeasure (collatzStep current)) decreased (current := collatzStep current)
            (path := path ++ [current])
            (env := setLocal (setLocal env "cm" (missValue limit current))
              "m" (.scalar (.int (collatzStep current))))
            (collatzStep_pos positive) locals.miss (.step trail positive notOne) rfl
        refine ⟨stop, visited, finalEnv, trace, stopPositive, finalLocals, lengthLocal, hit, ?_⟩
        apply Runtime.BlockContinuesIn.mono
          (small := 1 + (100 * (big.length + 1) +
            (500 * (big.length + 1) * (collatzLen (collatzStep current) + 1) + 1)) + 2)
        · apply Runtime.BlockContinuesIn.while_loop (cc := 1)
            (cb := 100 * (big.length + 1) +
              (500 * (big.length + 1) * (collatzLen (collatzStep current) + 1) + 1))
            Runtime.EvaluatesIn.constant (flag := true) rfl
          apply Runtime.BlockContinuesIn.append (ca := 100 * (big.length + 1))
            (cb := 500 * (big.length + 1) * (collatzLen (collatzStep current) + 1) + 1)
            (body := walkBody) (middleHeap := base ++ [.list (vertices (path ++ [current]))])
          · have advanced := walk_miss limit current env (base ++ [.list (vertices path)])
              smallAddress bigAddress base.length small (vertices path) big valid smallHere bigHere
              (by simp) locals.small locals.big locals.path locals.bound locals.current missing
            simpa [vertices, List.map_append] using advanced
          · exact Runtime.BlockContinuesIn.loop_resume (globals := []) recurse
        · rw [length_step current positive notOne]
          simp only [Nat.mul_add, Nat.mul_one]
          omega

end solvers.«14».runtimeCost
