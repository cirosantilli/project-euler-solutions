import solvers.«98.py.lean.d».SquareBounds
import data.«project-euler-statements».data.py.project_euler_statements.«p98.py»

namespace solvers.«98».pattern_signature
open FormalSW.PyAst HeapCalls

theorem «Returns» (text : String) :
    (HeapCalls.call program "pattern_signature" [.scalar (.string text)] []).Returns := by
  obtain ⟨_, _, returned, _⟩ := result text []
  exact returned.returns

theorem «Supported» (text : String) :
    (HeapCalls.call program "pattern_signature" [.scalar (.string text)] []).Supported :=
  («Returns» text).supported

/-- Two actual source calls yield equal tuple keys exactly when an injective substitution exists. -/
theorem pair_result (word digits : String) (heap : Mutable.Heap) :
    ∃ middle final left right,
      ReturnsValue program "pattern_signature" [.scalar (.string word)] heap (.ref left) middle ∧
      ReturnsValue program "pattern_signature" [.scalar (.string digits)] middle (.ref right) final ∧
      (final[left]? = final[right]? ↔
        ∃ f : Char → Char, word.toList.map f = digits.toList ∧
          Set.InjOn f {letter | letter ∈ word.toList}) := by
  obtain ⟨middle, left, first, firstFound, _⟩ := result word heap
  obtain ⟨final, right, second, secondFound, preserved⟩ := result digits middle
  obtain ⟨extra, rfl⟩ := preserved
  refine ⟨middle, middle ++ extra, left, right, first, second, ?_⟩
  have kept := ProjectEulerSolutions.HeapCollection.prefix_found middle extra left _ firstFound
  rw [kept, secondFound]
  have injective : Function.Injective number := by
    intro a b same
    have equal : (a : Int) = b := by simpa [number] using same
    omega
  simp only [Option.some.injEq, Mutable.Object.tuple.injEq]
  constructor
  · intro same
    have keys : ProjectEulerSolutions.AnagramSquares.labels [] word.toList =
        ProjectEulerSolutions.AnagramSquares.labels [] digits.toList :=
      (List.map_injective_iff.mpr injective) same
    exact (ProjectEulerSolutions.AnagramSquares.labels_eq_iff_substitution '0' _ _).mp keys
  · intro assignment
    have keys := (ProjectEulerSolutions.AnagramSquares.labels_eq_iff_substitution '0' _ _).mpr assignment
    simp [keys]

end solvers.«98».pattern_signature

namespace solvers.«98».solve
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.AnagramKeys
open pattern_signature (program)

abbrev keyExpression : Expr := .call (.attribute (.constant (.string "")) "join")
  [.builtinCall .sorted [.name "w"]]
abbrev grouping : Stmt := .forIn (.name "w") (.name "words")
  [.ifThen (.name "w")
    [.expr (.call (.attribute (.call (.attribute (.name "groups") "setdefault")
      [keyExpression, .list []]) "append") [.name "w"])] []] []

theorem source_grouping :
    (((findFunctionIn program.body "solve").getD ([], [])).2.drop 1).take 1 = [grouping] := rfl

theorem key_result (word : String) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (globals : Mutable.Globals) (getWord : env "w" = some (.scalar (.string word))) :
    Evaluates program env keyExpression heap (.scalar (.string (key word)))
      (heap ++ [.list (values word)]) globals :=
  evaluate_key program env heap (.name "w") word globals (Evaluates.lookup getWord)

/-- The reference accepts exactly the distinct words in the optimized dictionary's same key. -/
theorem reference_filter (word other : String) (address : Nat) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (globals : Mutable.Globals)
    (getWord : env "w" = some (.scalar (.string word)))
    (getOther : env "u" = some (.scalar (.string other)))
    (getKey : env "key" = some (.ref address))
    (found : heap[address]? = some (.list (values word))) :
    ∃ extra, Evaluates P98Ast.module env
      data.«project-euler-statements».data.py.project_euler_statements.p98.anagrams.condition heap
      (.scalar (.bool (decide (word ≠ other ∧ key word = key other)))) (heap ++ extra) globals :=
  data.«project-euler-statements».data.py.project_euler_statements.p98.anagrams.condition_result
    word other address env heap globals getWord getOther getKey found

end solvers.«98».solve
