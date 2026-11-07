import solvers.«98.py.lean.d».Setup
import ProjectEulerSolutions.Lib.PatternBucketLookup

namespace solvers.«98».candidates
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open pattern_signature (program)
open squares_with_length (signature squares)
set_option Elab.async false

def values (width : Nat) (word : String) : List String := (squares width).filter fun square => decide (signature square = signature word)
abbrev expression : Expr := .call (.attribute (.attribute (.name "sqinfo") "by_pattern") "get") [.name "p1", .list []]
abbrev groupBody : List Stmt := match (solve.source[9]? : Option Stmt) with | some (.forIn _ _ body _) => body | _ => []
abbrev wordBody : List Stmt := match (groupBody[2]? : Option Stmt) with | some (.forIn _ _ body _) => body | _ => []

theorem source : wordBody[1]? = some (.assign [.name "candidates"] expression) := rfl

/-- The source get call returns precisely the square strings with the word's canonical pattern. -/
theorem result (heap : Mutable.Heap) (outer : Nat) (width : Int) (recordAddress patternAddress : Nat) (word : String) (env : HeapCalls.Env)
    (stored : square_cache.Stored outer heap width recordAddress)
    (pattern : heap[patternAddress]? = some (word_patterns.tuple word))
    (getRecord : env "sqinfo" = some (.ref recordAddress)) (getPattern : env "p1" = some (.ref patternAddress)) :
    ∃ address, Evaluates program env expression heap (.ref address) (heap ++ [.list []]) ∧
      (heap ++ [.list []])[address]? = some (.list ((values width.toNat word).map PatternBuckets.text)) := by
  obtain ⟨dictionary, buckets, dictionaryRead, _, view, partition, _⟩ := square_cache.fields heap outer width recordAddress env stored getRecord
  refine ⟨PatternBuckets.chosen buckets (signature word) heap.length, ?_, ?_⟩
  · exact PatternBuckets.evaluate_get program env (.attribute (.name "sqinfo") "by_pattern") heap dictionary buckets view
      (signature word) patternAddress (by intros; simp_all) dictionaryRead getPattern pattern
  · have found := PatternBuckets.get_values heap dictionary buckets view (signature word)
    rw [partition.contents] at found
    exact found

theorem matched (width : Nat) (word square : String) (member : square ∈ values width word) :
    AnagramSquares.labels [] word.toList = AnagramSquares.labels [] square.toList := by
  exact (of_decide_eq_true (List.mem_filter.mp member).2).symm

theorem assignment (width : Nat) (word square : String) (member : square ∈ values width word) :
    ∃ f : Char → Char, word.toList.map f = square.toList ∧ Set.InjOn f {letter | letter ∈ word.toList} :=
  (AnagramSquares.labels_eq_iff_substitution '0' _ _).mp (matched width word square member)

end solvers.«98».candidates
