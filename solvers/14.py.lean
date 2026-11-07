import solvers.«14.py.lean.d».Source
import data.«project-euler-statements».data.py.project_euler_statements.«p14.py»
import FormalSW

namespace solvers.«14»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerStatements.P14

/-- The inclusive scalar interface agrees with the reference on nonpositive bounds. -/
theorem solve.nonpositive_result (limit : Int) (nonpositive : limit ≤ 0) (heap : Mutable.Heap) :
    ReturnsValue program "solve" [.scalar (.int limit)] heap (.scalar (.int 0)) heap := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.branch (value := .scalar (.bool true)) (flag := true) (middle := heap)
  · apply Evaluates.compare (Evaluates.lookup rfl) Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp, nonpositive]
  · rfl
  · exact BlockReturns.return_value Evaluates.constant

/-- The optimized bit test and arithmetic implement the statement's Collatz step. -/
private theorem advance_continues (n : Nat) (env : Mutable.Env) (heap : Mutable.Heap)
    (current : env "m" = some (.scalar (.int n))) :
    BlockContinues program env advanceBody heap
      (setLocal env "m" (.scalar (.int (collatzStep n)))) heap := by
  have test : Evaluates program env (.binOp (.name "m") .bitAnd (.constant (.int 1)))
      heap (.scalar (.int (↑(n % 2)))) heap := by
    apply Evaluates.binary (Evaluates.lookup current) Evaluates.constant
    simp [Mutable.binary, scalarBitAnd]
  by_cases even : n % 2 = 0
  · apply BlockContinues.branch test (flag := false)
      (by simp [Mutable.truth, scalarTruth, even])
    simp only [Bool.false_eq_true, ↓reduceIte, collatzStep, even, ↓reduceIte]
    apply BlockContinues.augAssign current Evaluates.constant
    simp [inplace, Mutable.binary, scalarBinary]
  · have odd : n % 2 = 1 := by omega
    apply BlockContinues.branch test (flag := true)
      (by simp [Mutable.truth, scalarTruth, odd])
    simp only [↓reduceIte, collatzStep, even, ↓reduceIte]
    apply BlockContinues.assign
    apply Evaluates.binary (av := .scalar (.int (3 * n))) (bv := .scalar (.int 1))
      (middle := heap) (afterRight := heap)
    · apply Evaluates.binary Evaluates.constant (Evaluates.lookup current)
      simp [Mutable.binary, scalarBinary]
    · exact Evaluates.constant
    · simp [Mutable.binary, scalarBinary]

/-- The actual unwinding branch stores a proved length in the appropriate cache. -/
private theorem store_continues (limit n : Nat) (env : Mutable.Env) (heap : Mutable.Heap)
    (smallAddress bigAddress : Nat) (small : List Mutable.Value) (big : List (Constant × Mutable.Value))
    (valid : Cache.Valid limit small big)
    (smallFound : heap[smallAddress]? = some (.list small))
    (bigFound : heap[bigAddress]? = some (.dict big))
    (smallLocal : env "cache_small" = some (.ref smallAddress))
    (bigLocal : env "cache_big" = some (.ref bigAddress))
    (limitLocal : env "limit" = some (.scalar (.int limit)))
    (vertexLocal : env "v" = some (.scalar (.int n)))
    (lengthLocal : env "length" = some (.scalar (.int (collatzLen n)))) :
    BlockContinues program env storeBody heap env
      (storedHeap limit n heap smallAddress bigAddress small big) := by
  have test : Evaluates program env (.compare (.name "v") [(.ltE, .name "limit")])
      heap (.scalar (.bool (decide (n ≤ limit)))) heap := by
    apply Evaluates.compare (Evaluates.lookup vertexLocal) (Evaluates.lookup limitLocal)
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  by_cases inside : n ≤ limit
  · apply BlockContinues.branch test (flag := true) (by simp [Mutable.truth, scalarTruth, inside])
    apply BlockContinues.setItem (by intro a b c; simp)
      (Evaluates.lookup lengthLocal)
      (.cons (Evaluates.lookup smallLocal) (.cons (Evaluates.lookup vertexLocal) .nil))
    simpa [storedHeap, inside] using
      Cache.write_small heap smallAddress n small (.scalar (.int (collatzLen n))) smallFound
        (by rw [valid.size]; omega)
  · apply BlockContinues.branch test (flag := false) (by simp [Mutable.truth, scalarTruth, inside])
    apply BlockContinues.setItem (by intro a b c; simp)
      (Evaluates.lookup lengthLocal)
      (.cons (Evaluates.lookup bigLocal) (.cons (Evaluates.lookup vertexLocal) .nil))
    simpa [storedHeap, inside] using
      Mutable.setItem_dictionary heap bigAddress big bigFound (.int n) (.scalar (.int (collatzLen n)))


/-- A cache hit exits the real search loop before appending or advancing the vertex. -/
private theorem walk_hit (limit n : Nat) (positive : 0 < n) (env : Mutable.Env) (heap : Mutable.Heap)
    (smallAddress bigAddress : Nat) (small : List Mutable.Value) (big : List (Constant × Mutable.Value))
    (valid : Cache.Valid limit small big)
    (smallFound : heap[smallAddress]? = some (.list small))
    (bigFound : heap[bigAddress]? = some (.dict big))
    (smallLocal : env "cache_small" = some (.ref smallAddress))
    (bigLocal : env "cache_big" = some (.ref bigAddress))
    (limitLocal : env "limit" = some (.scalar (.int limit)))
    (vertexLocal : env "m" = some (.scalar (.int n)))
    (known : if n ≤ limit then small[n]? = some (.scalar (.int (collatzLen n)))
      else ScalarDict.lookup big (.int n) = some (.scalar (.int (collatzLen n)))) :
    BlockContinues program env [walkLoop] heap
      (setLocal (setLocal env "cm" (.scalar (.int (collatzLen n)))) "length" (.scalar (.int (collatzLen n)))) heap := by
  have test : Evaluates program env (.compare (.name "m") [(.ltE, .name "limit")])
      heap (.scalar (.bool (decide (n ≤ limit)))) heap := by
    apply Evaluates.compare (Evaluates.lookup vertexLocal) (Evaluates.lookup limitLocal)
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  apply BlockContinues.while_loop Evaluates.constant rfl
  by_cases inside : n ≤ limit
  · apply BlockContinues.branch_next test (flag := true) (by simp [Mutable.truth, scalarTruth, inside])
    apply BlockContinues.append (body := [.assign [.name "cm"] (.subscript (.name "cache_small") (.name "m"))])
    · apply BlockContinues.assign (value := .scalar (.int (collatzLen n))) (finalHeap := heap)
      apply Evaluates.subscript (by intro a b c; simp) (Evaluates.lookup smallLocal) (Evaluates.lookup vertexLocal)
      have bound : n < small.length := by rw [valid.size]; omega
      have found := List.getElem?_eq_some_iff.mp (by simpa [inside] using known)
      simpa only [found.2] using Mutable.getItem_list heap smallAddress n small smallFound bound
    · apply BlockContinues.branch_next (Evaluates.lookup rfl) (flag := true)
        (by have nonzero := ProjectEulerStatements.Collatz.length_positive n positive
            simp [Mutable.truth, scalarTruth]; omega)
      apply BlockContinues.append (body := [.assign [.name "length"] (.name "cm")])
      · exact BlockContinues.assign (Evaluates.lookup rfl)
      · exact BlockContinues.break_loop rfl BlockContinues.nil
  · apply BlockContinues.branch_next test (flag := false) (by simp [Mutable.truth, scalarTruth, inside])
    apply BlockContinues.append (body := [.assign [.name "cm"] (.call (.attribute (.name "cache_big") "get") [.name "m"])])
    · apply BlockContinues.assign (value := .scalar (.int (collatzLen n))) (finalHeap := heap)
      apply Evaluates.dictionary_get (by intro name same; cases same; simp [bigLocal])
        (Evaluates.lookup bigLocal) bigFound (.cons (Evaluates.lookup vertexLocal) .nil)
      simp [primitive, Mutable.dictionaryGet, bigFound, show ScalarDict.lookup big (.int n) = some (.scalar (.int (collatzLen n))) by simpa [inside] using known]
    · apply BlockContinues.branch_next (value := .scalar (.bool true)) (flag := true)
      · apply Evaluates.unary
        · exact Evaluates.builtin (.cons (Evaluates.lookup rfl) (.cons .constant .nil)) rfl
        · rfl
      · rfl
      · apply BlockContinues.append (body := [.assign [.name "length"] (.name "cm")])
        · exact BlockContinues.assign (Evaluates.lookup rfl)
        · exact BlockContinues.break_loop rfl BlockContinues.nil

/-- A cache miss completes the probe and leaves the heap unchanged. -/
private theorem probe_miss (limit n : Nat) (env : Mutable.Env) (heap : Mutable.Heap)
    (smallAddress bigAddress : Nat) (small : List Mutable.Value) (big : List (Constant × Mutable.Value))
    (valid : Cache.Valid limit small big)
    (smallFound : heap[smallAddress]? = some (.list small))
    (bigFound : heap[bigAddress]? = some (.dict big))
    (smallLocal : env "cache_small" = some (.ref smallAddress))
    (bigLocal : env "cache_big" = some (.ref bigAddress))
    (limitLocal : env "limit" = some (.scalar (.int limit)))
    (vertexLocal : env "m" = some (.scalar (.int n)))
    (miss : if n ≤ limit then small[n]? = some (.scalar (.int 0))
      else ScalarDict.lookup big (.int n) = none) :
    BlockContinues program env probeBody heap (setLocal env "cm" (missValue limit n)) heap := by
  have test : Evaluates program env (.compare (.name "m") [(.ltE, .name "limit")])
      heap (.scalar (.bool (decide (n ≤ limit)))) heap := by
    apply Evaluates.compare (Evaluates.lookup vertexLocal) (Evaluates.lookup limitLocal)
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  by_cases inside : n ≤ limit
  · apply BlockContinues.branch test (flag := true) (by simp [Mutable.truth, scalarTruth, inside])
    apply BlockContinues.append (body := [.assign [.name "cm"] (.subscript (.name "cache_small") (.name "m"))])
    · apply BlockContinues.assign (value := .scalar (.int 0)) (finalHeap := heap)
      apply Evaluates.subscript (by intro a b c; simp) (Evaluates.lookup smallLocal) (Evaluates.lookup vertexLocal)
      have bound : n < small.length := by rw [valid.size]; omega
      have found := List.getElem?_eq_some_iff.mp (by simpa [inside] using miss)
      simpa only [found.2] using Mutable.getItem_list heap smallAddress n small smallFound bound
    · apply BlockContinues.branch (Evaluates.lookup rfl) (flag := false) rfl
      simpa [missValue, inside] using (BlockContinues.nil (module := program) (env := setLocal env "cm" (.scalar (.int 0))) (heap := heap))
  · apply BlockContinues.branch test (flag := false) (by simp [Mutable.truth, scalarTruth, inside])
    apply BlockContinues.append (body := [.assign [.name "cm"] (.call (.attribute (.name "cache_big") "get") [.name "m"])])
    · apply BlockContinues.assign (value := .scalar .none) (finalHeap := heap)
      apply Evaluates.dictionary_get (by intro name same; cases same; simp [bigLocal])
        (Evaluates.lookup bigLocal) bigFound (.cons (Evaluates.lookup vertexLocal) .nil)
      simp [primitive, Mutable.dictionaryGet, bigFound, show ScalarDict.lookup big (.int n) = none by simpa [inside] using miss]
    · apply BlockContinues.branch (value := .scalar (.bool false)) (flag := false)
      · apply Evaluates.unary
        · exact Evaluates.builtin (.cons (Evaluates.lookup rfl) (.cons .constant .nil)) rfl
        · rfl
      · rfl
      · simpa [missValue, inside] using (BlockContinues.nil (module := program) (env := setLocal env "cm" (.scalar .none)) (heap := heap))

private theorem walk_miss (limit n : Nat) (env : Mutable.Env) (heap : Mutable.Heap)
    (smallAddress bigAddress pathAddress : Nat) (small path : List Mutable.Value) (big : List (Constant × Mutable.Value))
    (valid : Cache.Valid limit small big)
    (smallFound : heap[smallAddress]? = some (.list small))
    (bigFound : heap[bigAddress]? = some (.dict big))
    (pathFound : heap[pathAddress]? = some (.list path))
    (smallLocal : env "cache_small" = some (.ref smallAddress))
    (bigLocal : env "cache_big" = some (.ref bigAddress))
    (pathLocal : env "path" = some (.ref pathAddress))
    (limitLocal : env "limit" = some (.scalar (.int limit)))
    (vertexLocal : env "m" = some (.scalar (.int n)))
    (miss : if n ≤ limit then small[n]? = some (.scalar (.int 0))
      else ScalarDict.lookup big (.int n) = none) :
    BlockContinues program env walkBody heap
      (setLocal (setLocal env "cm" (missValue limit n)) "m" (.scalar (.int (collatzStep n))))
      (heap.set pathAddress (.list (path ++ [.scalar (.int n)]))) := by
  apply BlockContinues.append (body := probeBody)
    (probe_miss limit n env heap smallAddress bigAddress small big valid smallFound bigFound smallLocal bigLocal limitLocal vertexLocal miss)
  apply BlockContinues.append (body := [.expr (.call (.attribute (.name "path") "append") [.name "m"])])
  · apply BlockContinues.discard
    exact Evaluates.append (Evaluates.lookup (by simpa [setLocal] using pathLocal)) pathFound
      (Evaluates.lookup (by simpa [setLocal] using vertexLocal)) pathFound
  · exact advance_continues n _ _ (by simpa [setLocal] using vertexLocal)

/-- The cache search terminates by the statement's existing Collatz measure.
Only the path object changes; all visited vertices are recorded in the trace. -/
private theorem walk_terminates (origin limit smallAddress bigAddress : Nat) (base : Mutable.Heap)
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
      BlockContinues program env [walkLoop] (base ++ [.list (vertices path)])
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
        · exact walk_hit limit current positive env _ smallAddress bigAddress small big valid smallHere bigHere
            locals.small locals.big locals.bound locals.current known
      · have notOne : current ≠ 1 := valid.miss_not_one current missing
        have decreased : collatzMeasure (collatzStep current) < measure := by
          rw [← h]
          exact collatzMeasure_decreases positive notOne
        obtain ⟨stop, visited, finalEnv, trace, stopPositive, finalLocals, lengthLocal, hit, recurse⟩ :=
          ih (collatzMeasure (collatzStep current)) decreased (current := collatzStep current)
            (path := path ++ [current]) (env := setLocal (setLocal env "cm" (missValue limit current)) "m" (.scalar (.int (collatzStep current))))
            (collatzStep_pos positive) locals.miss (.step trail positive notOne) rfl
        refine ⟨stop, visited, finalEnv, trace, stopPositive, finalLocals, lengthLocal, hit, ?_⟩
        apply BlockContinues.while_loop Evaluates.constant rfl
        apply BlockContinues.append (body := walkBody)
          (middleHeap := base ++ [.list (vertices (path ++ [current]))])
        · have advanced := walk_miss limit current env (base ++ [.list (vertices path)]) smallAddress bigAddress base.length
            small (vertices path) big valid smallHere bigHere (by simp)
            locals.small locals.big locals.path locals.bound locals.current missing
          simpa [vertices, List.map_append] using advanced
        · exact BlockContinues.loop_resume recurse

/-- One iteration restores the predecessor's length and writes it into its cache. -/
private theorem unwind_one (limit n : Nat) (positive : 0 < n) (notOne : n ≠ 1)
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
    BlockContinues program env unwindBody heap
      (setLocal env "length" (.scalar (.int (collatzLen n))))
      (storedHeap limit n heap smallAddress bigAddress small big) := by
  apply BlockContinues.append (body := [.augAssign (.name "length") .add (.constant (.int 1))])
    (middleEnv := setLocal env "length" (.scalar (.int (collatzLen n)))) (middleHeap := heap)
  · apply BlockContinues.augAssign lengthLocal Evaluates.constant
    simp [inplace, Mutable.binary, scalarBinary,
      ProjectEulerStatements.Collatz.length_step n positive notOne]
  · exact store_continues limit n _ heap smallAddress bigAddress small big valid smallFound bigFound
      (by simpa [setLocal] using smallLocal) (by simpa [setLocal] using bigLocal)
      (by simpa [setLocal] using limitLocal) (by simpa [setLocal] using vertexLocal) (by simp [setLocal])

/-- Unwind any recorded prefix, retaining the complete original path object.
Induction follows the recorded walk, so there is no fixed iteration bound. -/
private theorem unwind_prefix (origin current : Nat) (visitedPrefix : List Nat) (trail : Trail origin current visitedPrefix)
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
      BlockContinues program env
        [.reverseSequenceStepLoop "v" (.ref base.length) ((visitedPrefix.length : Int) - 1) unwindBody []]
        (base ++ [.list (vertices (visitedPrefix ++ suffix))]) finalEnv
        (finalBase ++ [.list (vertices (visitedPrefix ++ suffix))]) := by
  induction trail generalizing base small big suffix env with
  | start =>
      refine ⟨base, small, big, env, rfl, smallFound, bigFound, valid, known, locals, lengthLocal, ?_⟩
      exact BlockContinues.reverse_done (by decide) BlockContinues.nil
  | @step n path trace positive notOne ih =>
      have smallBound := (List.getElem?_eq_some_iff.mp smallFound).1
      have bigBound := (List.getElem?_eq_some_iff.mp bigFound).1
      let nextBase := storedHeap limit n base smallAddress bigAddress small big
      let nextEnv := setLocal (setLocal env "v" (.scalar (.int n))) "length" (.scalar (.int (collatzLen n)))
      have nextSize : nextBase.length = base.length := storedHeap_length limit n base smallAddress bigAddress small big
      obtain ⟨nextSmall, nextBig, nextSmallFound, nextBigFound, nextValid, nextKnown⟩ :=
        storedHeap_valid limit n base smallAddress bigAddress small big valid distinct smallFound bigFound
      have nextLocals : WalkLocals nextEnv origin limit smallAddress bigAddress nextBase.length stop := by
        simpa [nextEnv, nextSize] using locals.unwind n
      obtain ⟨finalBase, finalSmall, finalBig, finalEnv, finalSize, finalSmallFound, finalBigFound,
          finalValid, finalKnown, finalLocals, finalLength, recurse⟩ :=
        ih (base := nextBase) (small := nextSmall) (big := nextBig) (suffix := n :: suffix) (env := nextEnv)
          nextValid nextSmallFound nextBigFound nextLocals (by simp [nextEnv, setLocal]) nextKnown
      refine ⟨finalBase, finalSmall, finalBig, finalEnv, finalSize.trans nextSize,
        finalSmallFound, finalBigFound, finalValid, finalKnown, ?_, finalLength, ?_⟩
      · simpa only [nextSize] using finalLocals
      · simp only [List.length_append, List.length_singleton, Int.natCast_add, Int.natCast_one, Int.add_sub_cancel]
        apply BlockContinues.reverse_step (by omega)
          (items := vertices ((path ++ [n]) ++ suffix)) (value := .scalar (.int n))
          (by simp [sequenceValue, Mutable.reversibleSequence, Mutable.sequence])
          (by simp [vertices])
        apply BlockContinues.append (body := unwindBody) (middleEnv := nextEnv)
          (middleHeap := nextBase ++ [.list (vertices ((path ++ [n]) ++ suffix))])
        · have body := unwind_one limit n positive notOne (setLocal env "v" (.scalar (.int n)))
            (base ++ [.list (vertices ((path ++ [n]) ++ suffix))]) smallAddress bigAddress small big valid
            (by simpa [List.getElem?_append_left smallBound] using smallFound)
            (by simpa [List.getElem?_append_left bigBound] using bigFound)
            (by simpa [setLocal] using locals.small) (by simpa [setLocal] using locals.big)
            (by simpa [setLocal] using locals.bound) (by simp [setLocal]) (by simpa [setLocal] using lengthLocal)
          simpa only [storedHeap_append limit n base _ smallAddress bigAddress small big smallBound bigBound] using body
        · simpa [Calls.loopContinuation, unwindBody, nextSize, List.append_assoc] using recurse

private theorem length_return (limit n : Nat) (env : Mutable.Env) (heap : Mutable.Heap)
    (smallAddress bigAddress : Nat) (small : List Mutable.Value) (big : List (Constant × Mutable.Value))
    (valid : Cache.Valid limit small big)
    (smallFound : heap[smallAddress]? = some (.list small))
    (bigFound : heap[bigAddress]? = some (.dict big))
    (smallLocal : env "cache_small" = some (.ref smallAddress))
    (bigLocal : env "cache_big" = some (.ref bigAddress))
    (limitLocal : env "limit" = some (.scalar (.int limit)))
    (originLocal : env "n" = some (.scalar (.int n)))
    (known : Cache.Known limit small big n) :
    BlockReturns program env lengthReturnBody heap (.scalar (.int (collatzLen n))) heap := by
  have test : Evaluates program env (.compare (.name "n") [(.ltE, .name "limit")])
      heap (.scalar (.bool (decide (n ≤ limit)))) heap := by
    apply Evaluates.compare (Evaluates.lookup originLocal) (Evaluates.lookup limitLocal)
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  by_cases inside : n ≤ limit
  · apply BlockReturns.branch test (flag := true) (by simp [Mutable.truth, scalarTruth, inside])
    apply BlockReturns.return_value
    apply Evaluates.subscript (by intro a b c; simp) (Evaluates.lookup smallLocal) (Evaluates.lookup originLocal)
    have bound : n < small.length := by rw [valid.size]; omega
    have found := List.getElem?_eq_some_iff.mp (by simpa [Cache.Known, inside] using known)
    simpa only [found.2] using Mutable.getItem_list heap smallAddress n small smallFound bound
  · apply BlockReturns.branch test (flag := false) (by simp [Mutable.truth, scalarTruth, inside])
    apply BlockReturns.return_value
    apply Evaluates.subscript (by intro a b c; simp) (Evaluates.lookup bigLocal) (Evaluates.lookup originLocal)
    exact Mutable.getItem_dictionary heap bigAddress big bigFound (.int n) (by simpa [Cache.Known, inside] using known)

/-- The original memoizing helper terminates, returns the exact length, and
preserves both explicit caches while recording the requested result. -/
theorem collatz_length.result (limit n smallAddress bigAddress : Nat) (positive : 0 < n)
    (distinct : smallAddress ≠ bigAddress) (base : Mutable.Heap)
    (small : List Mutable.Value) (big : List (Constant × Mutable.Value))
    (valid : Cache.Valid limit small big)
    (smallFound : base[smallAddress]? = some (.list small))
    (bigFound : base[bigAddress]? = some (.dict big)) :
    ∃ finalHeap finalSmall finalBig,
      finalHeap[smallAddress]? = some (.list finalSmall) ∧
      finalHeap[bigAddress]? = some (.dict finalBig) ∧
      Cache.Valid limit finalSmall finalBig ∧ Cache.Known limit finalSmall finalBig n ∧
      ReturnsValue program "collatz_length" [.scalar (.int n), .scalar (.int limit), .ref smallAddress, .ref bigAddress]
        base (.scalar (.int (collatzLen n))) finalHeap := by
  let entry := functionEnv lengthDefinition.1 lengthDefinition.2
    [.scalar (.int n), .scalar (.int limit), .ref smallAddress, .ref bigAddress] []
  let initial := setLocal (setLocal entry "m" (.scalar (.int n))) "path" (.ref base.length)
  have initialLocals : WalkLocals initial n limit smallAddress bigAddress base.length n := by
    constructor <;> rfl
  have setup : BlockContinues program entry ((executableBody lengthDefinition.2).take 2) base
      initial (base ++ [.list []]) := by
    apply BlockContinues.append (body := [.assign [.name "m"] (.name "n")])
    · exact BlockContinues.assign (Evaluates.lookup rfl)
    · exact BlockContinues.annAssign (Evaluates.list .nil)
  obtain ⟨stop, path, afterWalk, trail, stopPositive, walkLocals, suffixLength, cachedStop, walked⟩ :=
    walk_terminates n limit smallAddress bigAddress base small big valid smallFound bigFound n [] initial positive initialLocals .start
  obtain ⟨finalBase, finalSmall, finalBig, finalEnv, finalSize, finalSmallFound, finalBigFound,
      finalValid, finalKnown, finalLocals, finalLength, unwound⟩ :=
    unwind_prefix n stop path trail limit smallAddress bigAddress stop distinct base small big [] afterWalk
      valid smallFound bigFound walkLocals suffixLength cachedStop
  have smallBound := (List.getElem?_eq_some_iff.mp finalSmallFound).1
  have bigBound := (List.getElem?_eq_some_iff.mp finalBigFound).1
  have smallHere : (finalBase ++ [.list (vertices path)])[smallAddress]? = some (.list finalSmall) := by
    simpa [List.getElem?_append_left smallBound] using finalSmallFound
  have bigHere : (finalBase ++ [.list (vertices path)])[bigAddress]? = some (.dict finalBig) := by
    simpa [List.getElem?_append_left bigBound] using finalBigFound
  have reversed : BlockContinues program afterWalk [.forReversed "v" (.name "path") unwindBody []]
      (base ++ [.list (vertices path)]) finalEnv (finalBase ++ [.list (vertices path)]) := by
    apply BlockContinues.for_reversed (items := vertices path) (Evaluates.lookup walkLocals.path)
      (by simp [Mutable.reversibleSequence, Mutable.sequence])
    simpa [vertices] using unwound
  refine ⟨finalBase ++ [.list (vertices path)], finalSmall, finalBig, smallHere, bigHere, finalValid, finalKnown, ?_⟩
  apply BlockReturns.function (args := lengthDefinition.1) (body := lengthDefinition.2) rfl rfl
  apply setup.returns
  apply walked.returns
  apply reversed.returns
  exact length_return limit n finalEnv _ smallAddress bigAddress finalSmall finalBig finalValid smallHere bigHere
    finalLocals.small finalLocals.big finalLocals.bound finalLocals.origin finalKnown

private theorem search_one (limit n smallAddress bigAddress winner : Nat) (positive : 0 < n)
    (distinct : smallAddress ≠ bigAddress) (env : Mutable.Env) (heap : Mutable.Heap)
    (small : List Mutable.Value) (big : List (Constant × Mutable.Value))
    (valid : Cache.Valid limit small big)
    (smallFound : heap[smallAddress]? = some (.list small))
    (bigFound : heap[bigAddress]? = some (.dict big))
    (locals : SearchLocals env limit smallAddress bigAddress winner) :
    ∃ finalHeap finalSmall finalBig finalEnv,
      finalHeap[smallAddress]? = some (.list finalSmall) ∧
      finalHeap[bigAddress]? = some (.dict finalBig) ∧
      Cache.Valid limit finalSmall finalBig ∧
      SearchLocals finalEnv limit smallAddress bigAddress
        (ProjectEulerStatements.ArgMaximum.select collatzLen winner n) ∧
      BlockContinues program (setLocal env "n" (.scalar (.int n))) searchBody heap finalEnv finalHeap := by
  obtain ⟨finalHeap, finalSmall, finalBig, finalSmallFound, finalBigFound, finalValid, _, called⟩ :=
    collatz_length.result limit n smallAddress bigAddress positive distinct heap small big valid smallFound bigFound
  let entry := setLocal env "n" (.scalar (.int n))
  let scored := setLocal entry "ln" (.scalar (.int (collatzLen n)))
  have evaluated : Evaluates program entry (.call (.name "collatz_length")
      [.name "n", .name "limit", .name "cache_small", .name "cache_big"])
      heap (.scalar (.int (collatzLen n))) finalHeap := by
    apply Evaluates.call (values := [.scalar (.int n), .scalar (.int limit), .ref smallAddress, .ref bigAddress])
    · exact .cons (Evaluates.lookup (by simp [entry, setLocal]))
        (.cons (Evaluates.lookup (by simpa [entry, setLocal] using locals.bound))
        (.cons (Evaluates.lookup (by simpa [entry, setLocal] using locals.small))
        (.cons (Evaluates.lookup (by simpa [entry, setLocal] using locals.big)) .nil)))
    · exact called
  have test : Evaluates program scored (.compare (.name "ln") [(.gt, .name "best_len")])
      finalHeap (.scalar (.bool (decide (collatzLen winner < collatzLen n)))) finalHeap := by
    apply Evaluates.compare (av := .scalar (.int (collatzLen n))) (Evaluates.lookup (by simp [scored, entry, setLocal]))
      (Evaluates.lookup (by simpa [scored, entry, setLocal] using locals.length))
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  by_cases better : collatzLen winner < collatzLen n
  · let finalEnv := setLocal (setLocal scored "best_len" (.scalar (.int (collatzLen n)))) "best_n" (.scalar (.int n))
    refine ⟨finalHeap, finalSmall, finalBig, finalEnv, finalSmallFound, finalBigFound, finalValid, ?_, ?_⟩
    · simp only [ProjectEulerStatements.ArgMaximum.select, if_pos better]
      constructor <;> simp [finalEnv, scored, entry, setLocal, locals.bound, locals.small, locals.big]
    · apply BlockContinues.append (body := [.assign [.name "ln"] (.call (.name "collatz_length")
          [.name "n", .name "limit", .name "cache_small", .name "cache_big"])])
        (middleEnv := scored) (middleHeap := finalHeap)
      · exact BlockContinues.assign evaluated
      · apply BlockContinues.branch test (flag := true) (by simp [Mutable.truth, scalarTruth, better])
        apply BlockContinues.append (body := [.assign [.name "best_len"] (.name "ln")])
          (middleEnv := setLocal scored "best_len" (.scalar (.int (collatzLen n)))) (middleHeap := finalHeap)
        · exact BlockContinues.assign (Evaluates.lookup (by simp [scored, setLocal]))
        · exact BlockContinues.assign (Evaluates.lookup (by simp [scored, entry, setLocal]))
  · refine ⟨finalHeap, finalSmall, finalBig, scored, finalSmallFound, finalBigFound, finalValid, ?_, ?_⟩
    · simp only [ProjectEulerStatements.ArgMaximum.select, if_neg better]
      constructor <;> simp [scored, entry, setLocal, locals.bound, locals.small, locals.big, locals.best, locals.length]
    · apply BlockContinues.append (body := [.assign [.name "ln"] (.call (.name "collatz_length")
          [.name "n", .name "limit", .name "cache_small", .name "cache_big"])])
        (middleEnv := scored) (middleHeap := finalHeap)
      · exact BlockContinues.assign evaluated
      · exact BlockContinues.branch test (flag := false) (by simp [Mutable.truth, scalarTruth, better]) BlockContinues.nil

private theorem search_loop (count start limit smallAddress bigAddress winner : Nat)
    (positive : 0 < start) (stop : start + count = limit) (distinct : smallAddress ≠ bigAddress)
    (env : Mutable.Env) (heap : Mutable.Heap)
    (small : List Mutable.Value) (big : List (Constant × Mutable.Value))
    (valid : Cache.Valid limit small big)
    (smallFound : heap[smallAddress]? = some (.list small))
    (bigFound : heap[bigAddress]? = some (.dict big))
    (locals : SearchLocals env limit smallAddress bigAddress winner) :
    ∃ finalHeap finalSmall finalBig finalEnv,
      finalHeap[smallAddress]? = some (.list finalSmall) ∧
      finalHeap[bigAddress]? = some (.dict finalBig) ∧
      Cache.Valid limit finalSmall finalBig ∧
      SearchLocals finalEnv limit smallAddress bigAddress
        ((List.range' start count).foldl (ProjectEulerStatements.ArgMaximum.select collatzLen) winner) ∧
      BlockContinues program env [.rangeStepLoop "n" start limit 1 searchBody []] heap finalEnv finalHeap := by
  induction count generalizing start winner env heap small big with
  | zero =>
      refine ⟨heap, small, big, env, smallFound, bigFound, valid, locals, ?_⟩
      apply BlockContinues.range_done (rest := BlockContinues.nil)
      simp [Calls.rangeContinues, ← stop]
  | succ count ih =>
      obtain ⟨middleHeap, middleSmall, middleBig, middleEnv, middleSmallFound, middleBigFound, middleValid,
          middleLocals, iteration⟩ := search_one limit start smallAddress bigAddress winner positive distinct
          env heap small big valid smallFound bigFound locals
      obtain ⟨finalHeap, finalSmall, finalBig, finalEnv, finalSmallFound, finalBigFound, finalValid, finalLocals, recurse⟩ :=
        ih (start + 1) (ProjectEulerStatements.ArgMaximum.select collatzLen winner start) (by omega) (by omega)
          middleEnv middleHeap middleSmall middleBig middleValid middleSmallFound middleBigFound middleLocals
      refine ⟨finalHeap, finalSmall, finalBig, finalEnv, finalSmallFound, finalBigFound, finalValid, ?_, ?_⟩
      · simpa only [List.range'_succ, List.foldl_cons] using finalLocals
      · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        apply BlockContinues.append iteration
        simpa [Calls.loopContinuation, searchBody] using recurse

/-- Construct both explicit caches by executing the source initialization. -/
private theorem search_initialize (limit : Nat) (positive : 0 < limit) :
    BlockContinues program (functionEnv searchDefinition.1 searchDefinition.2 [.scalar (.int limit)] [])
      ((executableBody searchDefinition.2).take 5) [] (searchEnv limit) (searchHeap limit) := by
  let entry := functionEnv searchDefinition.1 searchDefinition.2 [.scalar (.int limit)] []
  let allocated := setLocal entry "cache_small" (.ref 1)
  let zeros : List Mutable.Value := List.replicate (limit + 1) (.scalar (.int 0))
  let firstHeap : Mutable.Heap := [.list [.scalar (.int 0)], .list zeros]
  let initializedHeap : Mutable.Heap := [.list [.scalar (.int 0)], .list (Cache.initialSmall limit)]
  have allocatedList : Evaluates program entry (.binOp (.list [.constant (.int 0)]) .mult
      (.binOp (.name "limit") .add (.constant (.int 1)))) [] (.ref 1) firstHeap := by
    apply Evaluates.binary (av := .ref 0) (bv := .scalar (.int (limit + 1)))
      (middle := [.list [.scalar (.int 0)]]) (afterRight := [.list [.scalar (.int 0)]])
    · exact Evaluates.list (.cons Evaluates.constant .nil)
    · apply Evaluates.binary (Evaluates.lookup rfl) Evaluates.constant
      simp [Mutable.binary, scalarBinary]
    · change Mutable.allocate (List.replicate (limit + 1 : Int).toNat [.scalar (.int 0)]).flatten
        [.list [.scalar (.int 0)]] = _
      simp [Mutable.allocate, firstHeap, zeros, Except.map]
  have seed : BlockContinues program allocated [.assign [.subscript (.name "cache_small") (.constant (.int 1))]
      (.constant (.int 1))] firstHeap allocated initializedHeap := by
    apply BlockContinues.setItem (by intro a b c; simp) Evaluates.constant
      (.cons (Evaluates.lookup rfl) (.cons Evaluates.constant .nil))
    simpa [firstHeap, initializedHeap, Cache.initialSmall, zeros] using
      Cache.write_small firstHeap 1 1 zeros (.scalar (.int 1)) rfl (by simp [zeros]; omega)
  apply BlockContinues.append (body := [.assign [.name "cache_small"]
    (.binOp (.list [.constant (.int 0)]) .mult (.binOp (.name "limit") .add (.constant (.int 1))))])
    (middleEnv := allocated) (middleHeap := firstHeap)
  · exact BlockContinues.assign allocatedList
  · apply seed.append
    apply BlockContinues.append (body := [.annAssign (.name "cache_big")
      (.subscript (.name "dict") (.tuple [.name "int", .name "int"])) (some (.dict []))])
      (middleEnv := setLocal allocated "cache_big" (.ref 2)) (middleHeap := searchHeap limit)
    · exact BlockContinues.annAssign (Evaluates.dictionary .nil)
    · apply BlockContinues.append (body := [.assign [.name "best_n"] (.constant (.int 1))])
        (middleEnv := setLocal (setLocal allocated "cache_big" (.ref 2)) "best_n" (.scalar (.int 1)))
        (middleHeap := searchHeap limit)
      · exact BlockContinues.assign Evaluates.constant
      · exact BlockContinues.assign Evaluates.constant

/-- The exclusive-bound helper returns the first maximum and its length. -/
theorem longest_collatz_under.result (n : Nat) (positive : 0 < n) :
    ∃ heap, ReturnsValue program "longest_collatz_under" [.scalar (.int (n + 1))] []
      (.ref heap.length) (heap ++ [.tuple [.scalar (.int (ProjectEulerStatements.P14.naive n)),
        .scalar (.int (collatzLen (ProjectEulerStatements.P14.naive n)))]]) := by
  have initialLocals : SearchLocals (searchEnv (n + 1)) (n + 1) 1 2 1 := by
    refine ⟨rfl, rfl, rfl, rfl, ?_⟩
    simp [searchEnv, setLocal, ProjectEulerStatements.Collatz.length_one]
  obtain ⟨heap, small, big, env, smallFound, bigFound, valid, locals, scanned⟩ :=
    search_loop n 1 (n + 1) 1 2 1 (by omega) (by omega) (by decide) (searchEnv (n + 1))
      (searchHeap (n + 1)) (Cache.initialSmall (n + 1)) [] (Cache.initial_valid _ (by omega)) rfl rfl initialLocals
  rw [initial_winner n positive] at locals
  refine ⟨heap, ?_⟩
  apply BlockReturns.function (args := searchDefinition.1) (body := searchDefinition.2) rfl rfl
  apply (search_initialize (n + 1) (by omega)).returns
  apply BlockContinues.returns (body := [.forIn (.name "n") (.call (.name "range")
      [.constant (.int 1), .name "limit"]) searchBody []])
  · apply BlockContinues.for_range (.cons Evaluates.constant (.cons (Evaluates.lookup initialLocals.bound) .nil)) rfl
    exact scanned
  · apply BlockReturns.return_value
    exact Evaluates.tuple (.cons (Evaluates.lookup locals.best) (.cons (Evaluates.lookup locals.length) .nil)) (by simp)

/-- The public wrapper supplies the statement's inclusive bound and scalar answer. -/
theorem solve.positive_result (n : Nat) (positive : 0 < n) :
    ∃ heap, ReturnsValue program "solve" [.scalar (.int n)] []
      (.scalar (.int (ProjectEulerStatements.P14.naive n))) heap := by
  obtain ⟨heap, called⟩ := longest_collatz_under.result n positive
  refine ⟨heap ++ [.tuple [.scalar (.int (ProjectEulerStatements.P14.naive n)),
      .scalar (.int (collatzLen (ProjectEulerStatements.P14.naive n)))]], ?_⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.branch (value := .scalar (.bool false)) (flag := false) (middle := [])
  · apply Evaluates.compare (Evaluates.lookup rfl) Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega
  · rfl
  · apply BlockReturns.return_value
    apply Evaluates.subscript (by intro a b c; simp) (value := .ref heap.length)
    · apply Evaluates.call (values := [.scalar (.int (n + 1))]) (middle := [])
      · exact .cons (Evaluates.binary (middle := []) (afterRight := []) (Evaluates.lookup rfl) Evaluates.constant
          (by simp [Mutable.binary, scalarBinary])) .nil
      · exact called
    · exact Evaluates.constant
    · simp [Mutable.getItem, Mutable.index, Except.map]

theorem solve.«Equiv(naive)» (limit : Int) :
    (HeapCalls.scalarFunction program "solve" [.int limit]).Equiv
      (MemoCalls.sourceFunction P14Ast.module "naive" [.int limit]) := by
  by_cases positive : 0 < limit
  · lift limit to Nat using (by omega : 0 ≤ limit)
    obtain ⟨heap, executed⟩ := solve.positive_result limit (by omega)
    have optimized := ReturnsValue.scalar_finishes (values := [.int limit]) executed
    have reference := (data.«project-euler-statements».data.py.project_euler_statements.p14.Cached.naive_result limit).finishes (MemoCalls.Cache.valid_empty _)
    intro outcome
    constructor
    · intro finished
      rw [finished.unique optimized]
      exact reference
    · intro finished
      rw [finished.unique reference]
      exact optimized
  · have optimized := ReturnsValue.scalar_finishes (values := [.int limit])
        (solve.nonpositive_result limit (by omega) [])
    have reference : Finishes (MemoCalls.sourceFunction P14Ast.module "naive" [.int limit]).step
        (MemoCalls.sourceFunction P14Ast.module "naive" [.int limit]).initial (.ok (.int 0)) := by
      by_cases zero : limit = 0
      · subst limit
        simpa [ProjectEulerStatements.P14.naive, ProjectEulerStatements.ArgMaximum.argmax_range_succ] using
          (data.«project-euler-statements».data.py.project_euler_statements.p14.Cached.naive_result 0).finishes (MemoCalls.Cache.valid_empty _)
      · exact (data.«project-euler-statements».data.py.project_euler_statements.p14.Cached.naive_negative limit (by omega)).finishes (MemoCalls.Cache.valid_empty _)
    intro outcome
    constructor
    · intro finished
      rw [finished.unique optimized]
      exact reference
    · intro finished
      rw [finished.unique reference]
      exact optimized

theorem solve.«Supported» :
    (∀ limit : Int, (HeapCalls.scalarFunction program "solve" [.int limit]).Supported) ∧
    (∀ limit : Int, (MemoCalls.sourceFunction P14Ast.module "naive" [.int limit]).Supported) := by
  constructor
  · intro limit
    by_cases positive : 0 < limit
    · lift limit to Nat using (by omega : 0 ≤ limit)
      obtain ⟨heap, executed⟩ := solve.positive_result limit (by omega)
      exact (ReturnsValue.scalar_returns (values := [.int limit]) executed).supported
    · exact (ReturnsValue.scalar_returns (values := [.int limit]) (solve.nonpositive_result limit (by omega) [])).supported
  · exact data.«project-euler-statements».data.py.project_euler_statements.p14.naive.«Supported»

end solvers.«14»
