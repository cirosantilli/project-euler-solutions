import solvers.«14.py.lean.d».Source
import solvers.«14.py.lean.d».CacheWork
import lang.py.FormalSW.HeapRuntimeCostReverseLoops
import lang.py.FormalSW.HeapRuntimeCostLoopControl
import lang.py.FormalSW.HeapRuntimeCostRanges
import lang.py.FormalSW.HeapRuntimeCostListWrites
import lang.py.FormalSW.HeapRuntimeCostDictionaryWrites

/-! Runtime bounds for the actual cache writes during Collatz path unwinding. -/
namespace solvers.«14».runtimeCost
open FormalSW.PyAst FormalSW.PyAst.HeapCalls FormalSW.PyAst.HeapCalls.Runtime
open FormalSW.PyAst.PrimitiveWork
open ProjectEulerStatements.P14 ProjectEulerStatements.Collatz

/-- List writes are constant cost; dictionary writes retain collision-chain cost. -/
theorem store_continues (limit n : Nat) (env : Mutable.Env) (heap : Mutable.Heap)
    (smallAddress bigAddress : Nat) (small : List Mutable.Value) (big : List (Constant × Mutable.Value))
    (valid : Cache.Valid limit small big)
    (smallFound : heap[smallAddress]? = some (.list small))
    (bigFound : heap[bigAddress]? = some (.dict big))
    (smallLocal : env "cache_small" = some (.ref smallAddress))
    (bigLocal : env "cache_big" = some (.ref bigAddress))
    (limitLocal : env "limit" = some (.scalar (.int limit)))
    (vertexLocal : env "v" = some (.scalar (.int n)))
    (lengthLocal : env "length" = some (.scalar (.int (collatzLen n)))) :
    BlockContinuesIn .word program env storeBody heap (3 * big.length + 23) env
      (storedHeap limit n heap smallAddress bigAddress small big) := by
  have test : EvaluatesIn .word program env (.compare (.name "v") [(.ltE, .name "limit")])
      heap 7 (.scalar (.bool (decide (n ≤ limit)))) heap := by
    apply EvaluatesIn.compare (work := 2) (EvaluatesIn.lookup vertexLocal) (EvaluatesIn.lookup limitLocal)
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    · rfl
  by_cases inside : n ≤ limit
  · have written : BlockContinuesIn .word program env
        [.assign [.subscript (.name "cache_small") (.name "v")] (.name "length")]
        heap 13 env (storedHeap limit n heap smallAddress bigAddress small big) := by
      simpa only [storedHeap, if_pos inside] using
        BlockContinuesIn.list_setItem (globals := []) (arithmetic := .word) (module := program)
          (address := smallAddress) (offset := n) (position := n) (items := small)
          (afterRhs := heap) (afterArgs := heap)
          (by intro a b c equal; cases equal) (EvaluatesIn.lookup lengthLocal)
          (.cons (EvaluatesIn.lookup smallLocal) (.cons (EvaluatesIn.lookup vertexLocal) .nil))
          smallFound (Cache.write_small heap smallAddress n small _ smallFound (by rw [valid.size]; omega))
    exact (BlockContinuesIn.branch test (flag := true)
      (by simp [Mutable.truth, scalarTruth, inside]) written).mono (by omega)
  · have written : BlockContinuesIn .word program env
        [.assign [.subscript (.name "cache_big") (.name "v")] (.name "length")]
        heap (1 + 5 + mapLookup .collisionChain big.length (MapWork.keyWork .word big (.int n)) + 5)
        env (storedHeap limit n heap smallAddress bigAddress small big) := by
      simpa only [storedHeap, if_neg inside] using
        BlockContinuesIn.dictionary_setItem (globals := []) (arithmetic := .word) (module := program)
          (address := bigAddress) (key := .int n) (entries := big)
          (afterRhs := heap) (afterArgs := heap)
          (by intro a b c equal; cases equal) (EvaluatesIn.lookup lengthLocal)
          (.cons (EvaluatesIn.lookup bigLocal) (.cons (EvaluatesIn.lookup vertexLocal) .nil)) bigFound
    have bounded := valid.keyWork_word_le n
    exact (BlockContinuesIn.branch test (flag := false)
      (by simp [Mutable.truth, scalarTruth, inside]) written).mono (by
        simp only [mapLookup]
        have product := Nat.mul_le_mul_left (big.length + 1) (Nat.add_le_add_right bounded 1)
        simp only [Nat.mul_add, Nat.mul_one] at product ⊢
        omega)

/-- Increment the length and store it at the recorded vertex. -/
theorem unwind_one (limit n : Nat) (positive : 0 < n) (notOne : n ≠ 1)
    (env : Mutable.Env) (heap : Mutable.Heap) (smallAddress bigAddress : Nat)
    (small : List Mutable.Value) (big : List (Constant × Mutable.Value))
    (valid : Cache.Valid limit small big)
    (smallFound : heap[smallAddress]? = some (.list small))
    (bigFound : heap[bigAddress]? = some (.dict big))
    (smallLocal : env "cache_small" = some (.ref smallAddress))
    (bigLocal : env "cache_big" = some (.ref bigAddress))
    (limitLocal : env "limit" = some (.scalar (.int limit)))
    (vertexLocal : env "v" = some (.scalar (.int n)))
    (lengthLocal : env "length" = some (.scalar (.int (collatzLen (collatzStep n))))) :
    BlockContinuesIn .word program env unwindBody heap (3 * big.length + 30)
      (setLocal env "length" (.scalar (.int (collatzLen n))))
      (storedHeap limit n heap smallAddress bigAddress small big) := by
  have increment : BlockContinuesIn .word program env
      [.augAssign (.name "length") .add (.constant (.int 1))] heap 7
      (setLocal env "length" (.scalar (.int (collatzLen n)))) heap := by
    apply BlockContinuesIn.augAssign (work := 1) lengthLocal EvaluatesIn.constant
    · simp [inplace, Mutable.binary, scalarBinary, length_step n positive notOne]
    · rfl
  have write := store_continues limit n (setLocal env "length" (.scalar (.int (collatzLen n)))) heap smallAddress bigAddress small big valid smallFound bigFound
      (by simpa [setLocal] using smallLocal) (by simpa [setLocal] using bigLocal)
      (by simpa [setLocal] using limitLocal) (by simpa [setLocal] using vertexLocal) (by simp [setLocal])
  exact (increment.append write).mono (by omega)

theorem unwind_prefix (origin current : Nat) (visitedPrefix : List Nat) (trail : Trail origin current visitedPrefix)
    (limit smallAddress bigAddress stop : Nat) (distinct : smallAddress ≠ bigAddress)
    (base : Mutable.Heap) (small : List Mutable.Value) (big : List (Constant × Mutable.Value))
    (suffix : List Nat) (env : Mutable.Env)
    (valid : Cache.Valid limit small big)
    (smallFound : base[smallAddress]? = some (.list small))
    (bigFound : base[bigAddress]? = some (.dict big))
    (locals : WalkLocals env origin limit smallAddress bigAddress base.length stop)
    (lengthLocal : env "length" = some (.scalar (.int (collatzLen current))))
    (known : Cache.Known limit small big current) :
    ∃ finalBase finalSmall finalBig finalEnv,
      finalBase.length = base.length ∧
      finalBase[smallAddress]? = some (.list finalSmall) ∧
      finalBase[bigAddress]? = some (.dict finalBig) ∧
      Cache.Valid limit finalSmall finalBig ∧ Cache.Known limit finalSmall finalBig origin ∧
      WalkLocals finalEnv origin limit smallAddress bigAddress base.length stop ∧
      finalEnv "length" = some (.scalar (.int (collatzLen origin))) ∧
      finalBig.length ≤ big.length + visitedPrefix.length ∧
      BlockContinuesIn .word program env
        [.reverseSequenceStepLoop "v" (.ref base.length) ((visitedPrefix.length : Int) - 1) unwindBody []]
        (base ++ [.list (vertices (visitedPrefix ++ suffix))])
        (40 * (big.length + visitedPrefix.length + 1) * (visitedPrefix.length + 1)) finalEnv
        (finalBase ++ [.list (vertices (visitedPrefix ++ suffix))]) := by
  induction trail generalizing base small big suffix env with
  | start =>
      refine ⟨base, small, big, env, rfl, smallFound, bigFound, valid, known, locals, lengthLocal, by simp, ?_⟩
      exact (BlockContinuesIn.reverse_done (by decide) BlockContinuesIn.nil).mono (by simp [integer]; omega)
  | @step n path trace positive notOne ih =>
      have smallBound := (List.getElem?_eq_some_iff.mp smallFound).1
      have bigBound := (List.getElem?_eq_some_iff.mp bigFound).1
      let nextBase := storedHeap limit n base smallAddress bigAddress small big
      let nextEnv := setLocal (setLocal env "v" (.scalar (.int n))) "length" (.scalar (.int (collatzLen n)))
      have nextSize : nextBase.length = base.length := storedHeap_length limit n base smallAddress bigAddress small big
      obtain ⟨nextSmall, nextBig, nextSmallFound, nextBigFound, nextValid, nextKnown⟩ :=
        storedHeap_valid limit n base smallAddress bigAddress small big valid distinct smallFound bigFound
      have growth : nextBig.length ≤ big.length + 1 := by
        obtain ⟨updated, found, bounded⟩ := storedHeap_big_length_le limit n base
          smallAddress bigAddress small big distinct bigFound
        rw [nextBigFound] at found
        cases found
        exact bounded
      have nextLocals : WalkLocals nextEnv origin limit smallAddress bigAddress nextBase.length stop := by
        simpa [nextEnv, nextSize] using locals.unwind n
      obtain ⟨finalBase, finalSmall, finalBig, finalEnv, finalSize, finalSmallFound, finalBigFound,
          finalValid, finalKnown, finalLocals, finalLength, finalGrowth, recurse⟩ :=
        ih (base := nextBase) (small := nextSmall) (big := nextBig) (suffix := n :: suffix) (env := nextEnv)
          nextValid nextSmallFound nextBigFound nextLocals (by simp [nextEnv, setLocal]) nextKnown
      refine ⟨finalBase, finalSmall, finalBig, finalEnv, finalSize.trans nextSize,
        finalSmallFound, finalBigFound, finalValid, finalKnown, ?_, finalLength, ?_, ?_⟩
      · simpa only [nextSize] using finalLocals
      · simp only [List.length_append, List.length_singleton]
        omega
      · have body := unwind_one limit n positive notOne (setLocal env "v" (.scalar (.int n)))
            (base ++ [.list (vertices ((path ++ [n]) ++ suffix))]) smallAddress bigAddress small big valid
            (by simpa [List.getElem?_append_left smallBound] using smallFound)
            (by simpa [List.getElem?_append_left bigBound] using bigFound)
            (by simpa [setLocal] using locals.small) (by simpa [setLocal] using locals.big)
            (by simpa [setLocal] using locals.bound) (by simp [setLocal]) (by simpa [setLocal] using lengthLocal)
        have body' : BlockContinuesIn .word program (setLocal env "v" (.scalar (.int n))) unwindBody
            (base ++ [.list (vertices ((path ++ [n]) ++ suffix))]) (3 * big.length + 30)
            nextEnv (nextBase ++ [.list (vertices ((path ++ [n]) ++ suffix))]) := by
          simpa only [storedHeap_append limit n base _ smallAddress bigAddress small big smallBound bigBound] using body
        have rest : BlockContinuesIn .word program nextEnv
            [Calls.loopContinuation unwindBody (.reverseSequenceStepLoop "v" (.ref base.length)
              ((path.length : Int) - 1) unwindBody [])]
            (nextBase ++ [.list (vertices ((path ++ [n]) ++ suffix))])
            (40 * (nextBig.length + path.length + 1) * (path.length + 1)) finalEnv
            (finalBase ++ [.list (vertices ((path ++ [n]) ++ suffix))]) := by
          simpa [Calls.loopContinuation, unwindBody, nextSize, List.append_assoc] using recurse
        have step := BlockContinuesIn.reverse_list_step (arithmetic := .word)
          (position := (path.length : Int)) (by omega)
          (address := base.length) (items := vertices ((path ++ [n]) ++ suffix))
          (by simp) (value := .scalar (.int n)) (by simp [vertices]) (body'.append rest)
        have costBound : 3 * big.length + 30 +
            40 * (nextBig.length + path.length + 1) * (path.length + 1) + 5 ≤
            40 * (big.length + (path.length + 1) + 1) * ((path.length + 1) + 1) := by
          have monotone := Nat.mul_le_mul_right (path.length + 1)
            (Nat.mul_le_mul_left 40 (show nextBig.length + path.length + 1 ≤
              big.length + (path.length + 1) + 1 by omega))
          have expand : 40 * (big.length + (path.length + 1) + 1) * ((path.length + 1) + 1) =
              40 * (big.length + (path.length + 1) + 1) * (path.length + 1) +
              40 * (big.length + (path.length + 1) + 1) := by rw [Nat.mul_add, Nat.mul_one]
          rw [expand]
          omega
        simpa only [List.length_append, List.length_singleton, Int.natCast_add, Int.natCast_one,
          Int.add_sub_cancel] using step.mono costBound

end solvers.«14».runtimeCost
