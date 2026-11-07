import solvers.«98.py.lean.d».CacheLookup
import solvers.«98.py.lean.d».WordCollect

namespace solvers.«98».solve
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open pattern_signature (program number)
set_option Elab.async false

abbrev source : List Stmt := ((findFunctionIn program.body "solve").getD ([], [])).2
abbrev initial (address : Nat) : HeapCalls.Env :=
  functionEnv ((findFunctionIn program.body "solve").getD ([], [])).1 source [.ref address] []

/-- Source-visible objects available when the optimized search begins. -/
structure Ready (words : List String) (heap : Mutable.Heap) (env : HeapCalls.Env) where
  buckets : List StringValueBuckets.Bucket
  groupDictionary : Nat
  groupAddress : Nat
  squareAddress : Nat
  patternAddress : Nat
  squareRows : square_cache.Rows
  patternRows : word_patterns.Rows
  groupView : StringValueBuckets.View heap groupDictionary buckets
  partition : StringValueBuckets.Partition keyOfValue (groupValues words) buckets
  groupList : heap[groupAddress]? = some (.list ((groups.selected buckets).map groups.reference))
  squareDictionary : heap[squareAddress]? = some (.dict (KeyedReferences.entries KeyedReferences.integers squareRows))
  squareValid : square_cache.Valid squareAddress heap squareRows
  squareCoverage : ∀ width, (KeyedReferences.lookup squareRows width).isSome ↔ width ∈ lengths.needed words buckets
  patternDictionary : heap[patternAddress]? = some (.dict (KeyedReferences.entries KeyedReferences.strings patternRows))
  patternValid : word_patterns.Valid heap patternRows
  patternCoverage : ∀ word, (KeyedReferences.lookup patternRows word).isSome ↔ word ∈ word_patterns.allWords words (groups.selected buckets)
  getGroups : env "anagram_groups" = some (.ref groupAddress)
  getSquares : env "squares_cache" = some (.ref squareAddress)
  getPatterns : env "word_pat" = some (.ref patternAddress)
  getBest : env "best" = some (number 0)

/-- The complete setup either returns the correct zero immediately, or reaches the search with both exact caches. -/
theorem setup (words : List String) (heap : Mutable.Heap) (wordsAddress : Nat)
    (found : heap[wordsAddress]? = some (.list (words.map text))) :
    (∃ extra, ReturnsValue program "solve" [.ref wordsAddress] heap (number 0) (heap ++ extra) ∧ AnagramSearch.solve words = 0) ∨
    (∃ extra env, BlockContinues program (initial wordsAddress) (source.take 9) heap env (heap ++ extra) ∧ Nonempty (Ready words (heap ++ extra) env)) := by
  obtain ⟨groupEnv, groupExtra, buckets, groupAddress, grouped, groupView, partition, groupList, _, getGroups, _⟩ :=
    groups.prefix_result words heap wordsAddress (initial wordsAddress) [] found (by rfl)
  by_cases empty : groups.selected buckets = []
  · refine Or.inl ⟨groupExtra, ?_, ?_⟩
    · apply BlockReturns.function rfl rfl
      exact grouped.returns (groups.empty_branch buckets (heap ++ groupExtra) groupAddress groupEnv [] empty groupList getGroups (source.drop 4))
    · simp [AnagramSearch.solve, groups.no_pairs words buckets partition empty, AnagramSearch.scanPairs]
  · have guarded := groups.nonempty_guard buckets (heap ++ groupExtra) groupAddress groupEnv [] empty groupList getGroups
    let groupHeap := heap ++ groupExtra
    let lengthExtra : Mutable.Heap := [.set ((lengths.collect [] (lengths.widths words (groups.selected buckets))).map Constant.int),
      .list ((lengths.needed words buckets).map lengths.number)]
    let lengthHeap := groupHeap ++ lengthExtra
    let lengthEnv := setLocal groupEnv "lengths_needed" (.ref (groupHeap.length + 1))
    have measured := lengths.result words buckets groupHeap heap.length groupAddress groupEnv [] groupView partition groupList getGroups
    have lengthList : lengthHeap[groupHeap.length + 1]? = some (.list ((lengths.needed words buckets).map lengths.number)) := by
      simp [lengthHeap, lengthExtra]
    obtain ⟨squareRows, squareGarbage, squareBuilt, squareValid, squareCoverage⟩ :=
      square_cache.result (lengths.needed words buckets) (lengths.positive words buckets) lengthHeap (groupHeap.length + 1) lengthEnv
        lengthList (by simp [lengthEnv, setLocal])
    let squareExtra : Mutable.Heap := .dict (KeyedReferences.entries KeyedReferences.integers squareRows) :: squareGarbage
    let squareHeap := lengthHeap ++ squareExtra
    let squareEnv := setLocal lengthEnv "squares_cache" (.ref lengthHeap.length)
    have squareGroupView : StringValueBuckets.View squareHeap heap.length buckets := (groupView.append lengthExtra).append squareExtra
    have squareGroupList : squareHeap[groupAddress]? = some (.list ((groups.selected buckets).map groups.reference)) :=
      HeapCollection.prefix_found lengthHeap squareExtra groupAddress _ (HeapCollection.prefix_found groupHeap lengthExtra groupAddress _ groupList)
    have squareGetGroups : squareEnv "anagram_groups" = some (.ref groupAddress) := by
      simpa [squareEnv, lengthEnv, setLocal] using getGroups
    obtain ⟨wordEnv, patternRows, patternGarbage, wordBuilt, patternValid, patternCoverage, getPatterns, wordKept⟩ :=
      word_patterns.result words buckets squareHeap heap.length groupAddress squareEnv squareGroupView partition squareGroupList squareGetGroups
    let patternExtra : Mutable.Heap := .dict (KeyedReferences.entries KeyedReferences.strings patternRows) :: patternGarbage
    let finalHeap := squareHeap ++ patternExtra
    let finalEnv := setLocal wordEnv "best" (number 0)
    have prepared : Ready words finalHeap finalEnv := by
      refine ⟨buckets, heap.length, groupAddress, lengthHeap.length, squareHeap.length, squareRows, patternRows,
        squareGroupView.append patternExtra, partition, ?_, ?_, ?_, squareCoverage, ?_, patternValid, patternCoverage, ?_, ?_, ?_, ?_⟩
      · exact HeapCollection.prefix_found squareHeap patternExtra groupAddress _ squareGroupList
      · exact HeapCollection.prefix_found squareHeap patternExtra lengthHeap.length _
          (KeyedReferences.found KeyedReferences.integers lengthHeap squareRows squareGarbage)
      · exact squareValid.append patternExtra
      · exact KeyedReferences.found KeyedReferences.strings squareHeap patternRows patternGarbage
      · have kept := (wordKept "anagram_groups" (by decide)).trans squareGetGroups
        simpa [finalEnv, setLocal] using kept
      · have kept := (wordKept "squares_cache" (by decide)).trans (show squareEnv "squares_cache" = some (.ref lengthHeap.length) by simp [squareEnv, setLocal])
        simpa [finalEnv, setLocal] using kept
      · simpa [finalEnv, setLocal] using getPatterns
      · simp [finalEnv, setLocal]
    refine Or.inr ⟨groupExtra ++ lengthExtra ++ squareExtra ++ patternExtra, finalEnv, ?_, ?_⟩
    · have collected : BlockContinues program groupEnv [.assign [.name "lengths_needed"] lengths.sortedExpression]
          groupHeap lengthEnv lengthHeap := by
        simpa only [lengthHeap, lengthExtra, lengths.work, List.append_assoc, List.singleton_append] using BlockContinues.assign measured
      have cached : BlockContinues program lengthEnv (source.drop 5 |>.take 1) lengthHeap squareEnv squareHeap :=
        BlockContinues.annAssign squareBuilt
      have started : BlockContinues program wordEnv [.assign [.name "best"] (.constant (.int 0))] finalHeap finalEnv finalHeap :=
        BlockContinues.assign Evaluates.constant
      have executed := grouped.append (guarded.append (collected.append (cached.append (wordBuilt.append started))))
      simpa only [source, finalHeap, squareHeap, lengthHeap, groupHeap, List.append_assoc] using executed
    · simpa only [finalHeap, squareHeap, lengthHeap, groupHeap, List.append_assoc] using (show Nonempty (Ready words finalHeap finalEnv) from ⟨prepared⟩)

end solvers.«98».solve
