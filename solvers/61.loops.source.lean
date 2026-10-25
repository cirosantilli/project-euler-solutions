import solvers.«61.arguments.source»

set_option Elab.async false

namespace solvers.«61».dfs
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
private abbrev program := Solvers.P61Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)

def advance (values : List Int) (visit : Int → Int) (best value : Int) : Int :=
  if value ∈ values then best else max best (visit value)

/-- The indexed candidate row computes a maximum while preserving every captured source object. -/
theorem candidate_loop (state : SearchState heap path remainingAddress prefixCell maxSides values remaining)
    (bound : Bindings env path remainingAddress prefixCell) (kind : Int) (unused : Nat)
    (unusedFound : heap[unused]? = some (.list (kinds (remaining.filter (fun other => decide (other ≠ kind))))))
    (getUnused : env "unused" = some (.ref unused)) (best : Int) (getBest : env "best" = some (number best))
    (all : List Int) (source : Nat) (found : heap[source]? = some (.list (numbers all))) (visit : Int → Int)
    (child : ∀ currentHeap value, SearchState currentHeap path remainingAddress prefixCell maxSides values remaining →
      currentHeap[unused]? = some (.list (kinds (remaining.filter (fun other => decide (other ≠ kind))))) →
      value ∈ all → value ∉ values →
      ∃ extra, ReturnsValue program "__formalsw_local_solve_dfs"
        [.ref (currentHeap.length + 1), .ref unused, .ref prefixCell] (childHeap currentHeap values value)
        (number (visit value)) (childHeap currentHeap values value ++ extra) globals) :
    ∃ extra finalEnv, BlockContinues program env [.sequenceStepLoop "nxt" (.ref source) 0 innerBody []]
      heap finalEnv (heap ++ extra) globals ∧ Bindings finalEnv path remainingAddress prefixCell ∧
      finalEnv "best" = some (number (all.foldl (advance values visit) best)) ∧
      HeapRangeFold.Frame ["nxt", "best"] env finalEnv := by
  let invariant := fun currentHeap currentEnv currentBest =>
    Nonempty (SearchState currentHeap path remainingAddress prefixCell maxSides values remaining) ∧
      Bindings currentEnv path remainingAddress prefixCell ∧ currentEnv "unused" = some (.ref unused) ∧
      currentHeap[unused]? = some (.list (kinds (remaining.filter (fun other => decide (other ≠ kind))))) ∧
      currentEnv "best" = some (number currentBest)
  have bindInvariant : ∀ currentHeap currentEnv currentBest value, invariant currentHeap currentEnv currentBest → value ∈ all →
      invariant currentHeap (setLocal currentEnv "nxt" (number value)) currentBest := by
    intro currentHeap currentEnv currentBest value good _
    obtain ⟨ready, currentBound, currentUnused, currentFound, currentBestFound⟩ := good
    refine ⟨ready, currentBound.local "nxt" (number value) (by decide), ?_, currentFound, ?_⟩
    · simpa [setLocal] using currentUnused
    · simpa [setLocal] using currentBestFound
  have cell : ∀ currentHeap currentEnv currentBest value, invariant currentHeap currentEnv currentBest → value ∈ all →
      currentEnv "nxt" = some (number value) →
      ∃ extra finalEnv, HeapLoop.Continues program currentEnv innerBody currentHeap finalEnv (currentHeap ++ extra) globals ∧
        invariant (currentHeap ++ extra) finalEnv (advance values visit currentBest value) ∧
        HeapRangeFold.Frame ["nxt", "best"] currentEnv finalEnv := by
    intro currentHeap currentEnv currentBest value good member getValue
    obtain ⟨⟨current⟩, currentBound, currentUnused, currentFound, currentBestFound⟩ := good
    have unchanged : invariant currentHeap currentEnv currentBest :=
      ⟨⟨current⟩, currentBound, currentUnused, currentFound, currentBestFound⟩
    by_cases used : value ∈ values
    · refine ⟨[], currentEnv, ?_, ?_, HeapRangeFold.Frame.refl _ _⟩
      · simpa only [List.append_nil] using inner_used values currentHeap path current.pathFound value used currentEnv
          currentBound.getPath getValue globals
      · simpa only [advance, if_pos used, List.append_nil] using unchanged
    · obtain ⟨extra, returned⟩ := child currentHeap value current currentFound member used
      let total : Mutable.Heap := [.list [number value], .list (numbers (values ++ [value]))] ++ extra
      have layout : childHeap currentHeap values value ++ extra = currentHeap ++ total := by
        simp only [childHeap, total, List.append_assoc]
      rw [layout] at returned
      let finalEnv := setLocal currentEnv "best" (number (max currentBest (visit value)))
      have finalGood : invariant (currentHeap ++ total) finalEnv (max currentBest (visit value)) := by
        refine ⟨⟨current.append total⟩, currentBound.local "best" _ (by decide), ?_, ?_, rfl⟩
        · simpa [finalEnv, setLocal] using currentUnused
        · exact HeapCollection.prefix_found _ _ _ _ currentFound
      refine ⟨total, finalEnv, ?_, ?_, HeapRangeFold.Frame.updated _ currentEnv "best" _ (by decide)⟩
      · exact inner_updated values currentHeap path unused prefixCell current.pathFound value currentBest
          (visit value) used currentEnv currentBound.getPath getValue currentUnused currentBestFound currentBound.getCell returned
      · simpa only [advance, if_neg used] using finalGood
  obtain ⟨extra, finalEnv, executed, good, frame⟩ := HeapLoop.fold program "nxt" innerBody rfl ["nxt", "best"] (by decide)
    number (advance values visit) invariant source all bindInvariant cell all 0 rfl heap env best
    ⟨⟨state⟩, bound, getUnused, unusedFound, getBest⟩ found
  exact ⟨extra, finalEnv, executed, good.2.1, good.2.2.2.2, frame⟩

def advanceKind (values : List Int) (suffix : Int) (visit : Int → Int → Int) (best kind : Int) : Int :=
  (PolygonalCycleSearch.indexed PolygonalFamilies.family kind suffix).foldl (advance values (visit kind)) best

/-- Type traversal includes the fresh unused list and the eager default allocation for every lookup. -/
theorem type_loop (state : SearchState heap path remainingAddress prefixCell maxSides values remaining)
    (bound : Bindings env path remainingAddress prefixCell) (suffix : Int)
    (getSuffix : env "suffix" = some (number suffix))
    (best : Int) (getBest : env "best" = some (number best)) (visit : Int → Int → Int)
    (child : ∀ currentHeap kind unused value, SearchState currentHeap path remainingAddress prefixCell maxSides values remaining →
      kind ∈ remaining →
      currentHeap[unused]? = some (.list (kinds (remaining.filter (fun other => decide (other ≠ kind))))) →
      value ∈ PolygonalCycleSearch.indexed PolygonalFamilies.family kind suffix → value ∉ values →
      ∃ extra, ReturnsValue program "__formalsw_local_solve_dfs"
        [.ref (currentHeap.length + 1), .ref unused, .ref prefixCell] (childHeap currentHeap values value)
        (number (visit kind value)) (childHeap currentHeap values value ++ extra) globals) :
    ∃ extra finalEnv, BlockContinues program env [.forIn (.name "t") (.name "remaining") outerBody []]
      heap finalEnv (heap ++ extra) globals ∧ Bindings finalEnv path remainingAddress prefixCell ∧
      finalEnv "best" = some (number (remaining.foldl (advanceKind values suffix visit) best)) ∧
      HeapRangeFold.Frame ["t", "unused", "nxt", "best"] env finalEnv := by
  let invariant := fun currentHeap currentEnv currentBest =>
    Nonempty (SearchState currentHeap path remainingAddress prefixCell maxSides values remaining) ∧
      Bindings currentEnv path remainingAddress prefixCell ∧ currentEnv "suffix" = some (number suffix) ∧
      currentEnv "best" = some (number currentBest)
  have bindInvariant : ∀ currentHeap currentEnv currentBest kind, invariant currentHeap currentEnv currentBest → kind ∈ remaining →
      invariant currentHeap (setLocal currentEnv "t" (number kind)) currentBest := by
    intro currentHeap currentEnv currentBest kind good _
    obtain ⟨ready, currentBound, currentSuffix, currentBestFound⟩ := good
    exact ⟨ready, currentBound.local "t" _ (by decide), by simpa [setLocal] using currentSuffix,
      by simpa [setLocal] using currentBestFound⟩
  have cell : ∀ currentHeap currentEnv currentBest kind, invariant currentHeap currentEnv currentBest → kind ∈ remaining →
      currentEnv "t" = some (number kind) →
      ∃ afterEnv finalHeap, BlockContinues program currentEnv outerBody currentHeap afterEnv finalHeap globals ∧
        invariant finalHeap afterEnv (advanceKind values suffix visit currentBest kind) ∧
        (∃ extra, finalHeap = currentHeap ++ extra) ∧
        HeapRangeFold.Frame ["t", "unused", "nxt", "best"] currentEnv afterEnv := by
    intro currentHeap currentEnv currentBest kind good member getKind
    obtain ⟨⟨current⟩, currentBound, currentSuffix, currentBestFound⟩ := good
    let filtered := remaining.filter (fun other => decide (other ≠ kind))
    let middle := currentHeap ++ [.list (kinds filtered)]
    let readyEnv := setLocal currentEnv "unused" (.ref currentHeap.length)
    let ready := current.append [.list (kinds filtered)]
    have readyBound : Bindings readyEnv path remainingAddress prefixCell := currentBound.local "unused" _ (by decide)
    have readyBest : readyEnv "best" = some (number currentBest) := by simpa [readyEnv, setLocal] using currentBestFound
    have readyKind : readyEnv "t" = some (number kind) := by simpa [readyEnv, setLocal] using getKind
    have readySuffix : readyEnv "suffix" = some (number suffix) := by simpa [readyEnv, setLocal] using currentSuffix
    have made := unused_evaluate remaining kind currentHeap remainingAddress current.remainingFound currentEnv
      currentBound.getRemaining getKind globals
    obtain ⟨source, looked, found⟩ := next_values_evaluate ready.world kind suffix
      (current.validKinds kind member) readyEnv readyBound.getCell readyKind readySuffix globals
    let searched := ready.append [.list []]
    have unusedFound : (middle ++ [.list []])[currentHeap.length]? = some (.list (kinds filtered)) := by simp [middle]
    obtain ⟨extra, finalEnv, scanned, finalBound, finalBest, frame⟩ := candidate_loop searched readyBound kind currentHeap.length
      unusedFound rfl currentBest readyBest (PolygonalCycleSearch.indexed PolygonalFamilies.family kind suffix) source found
      (visit kind) (fun nextHeap value ready found inFamily fresh =>
        child nextHeap kind currentHeap.length value ready member found inFamily fresh)
    have sequenceFound : (middle ++ [.list []])[source]? =
        some (.list (numbers (PolygonalCycleSearch.indexed PolygonalFamilies.family kind suffix))) := found
    simp only [middle, List.append_assoc, List.singleton_append] at sequenceFound
    have header : BlockContinues program readyEnv [.forIn (.name "nxt") nextValues innerBody []]
        middle finalEnv ((middle ++ [.list []]) ++ extra) globals := by
      apply BlockContinues.for_sequence
        (items := numbers (PolygonalCycleSearch.indexed PolygonalFamilies.family kind suffix))
        (by intro expressions impossible; cases impossible) looked (by simp [Mutable.sequence, sequenceFound])
      exact scanned
    have emitted := (BlockContinues.assign (name := "unused") made).append header
    let total : Mutable.Heap := .list (kinds filtered) :: .list [] :: extra
    have layout : ((middle ++ [.list []]) ++ extra) = currentHeap ++ total := by
      simp [middle, total, List.append_assoc]
    rw [layout] at emitted
    have widened : HeapRangeFold.Frame ["t", "unused", "nxt", "best"] readyEnv finalEnv := by
      intro name outside
      exact frame name (fun member => outside (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ member)))
    refine ⟨finalEnv, currentHeap ++ total, emitted, ⟨⟨current.append total⟩, finalBound, ?_, finalBest⟩,
      ⟨total, rfl⟩, ?_⟩
    · exact (frame "suffix" (by decide)).trans readySuffix
    · exact (HeapRangeFold.Frame.updated _ currentEnv "unused" (.ref currentHeap.length) (by decide)).trans widened
  obtain ⟨finalEnv, finalHeap, executed, good, ⟨extra, layout⟩, frame⟩ := HeapSequenceFold.for_list program "t" outerBody rfl
    ["t", "unused", "nxt", "best"] (by decide) (fun kind : Int => number kind)
    (advanceKind values suffix visit) invariant remainingAddress remaining bindInvariant cell heap env best
    ⟨⟨state⟩, bound, getSuffix, getBest⟩ (.name "remaining") (by intro expressions impossible; cases impossible)
    (Evaluates.lookup bound.getRemaining) state.remainingFound
  rw [layout] at executed
  exact ⟨extra, finalEnv, executed, good.2.1, good.2.2.2, frame⟩

end solvers.«61».dfs
