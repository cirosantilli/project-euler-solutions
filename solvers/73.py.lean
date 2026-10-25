import solvers.«73.source»
import data.«project-euler-statements».data.py.project_euler_statements.«p73.py»

/-! Structural source termination, equivalence, and every trailing-default call shape. -/
namespace solvers.«73»
open FormalSW.PyAst ProjectEulerStatements.FractionIntervalCount
open data.«project-euler-statements».data.py.project_euler_statements.p73
  (denominatorCondition denominator_positive values naiveArgs)
set_option Elab.async false

abbrev reference := P73Ast.module

theorem count_between.«Returns» (d lowerNum lowerDen upperNum upperDen : Int)
    (lo : denominatorCondition program "lower_den" lowerDen)
    (hi : denominatorCondition program "upper_den" upperDen) :
    (Calls.function program "count_between"
      (values d ⟨lowerNum, lowerDen, upperNum, upperDen⟩)).Returns :=
  (count_between.result d ⟨lowerNum, lowerDen, upperNum, upperDen⟩
    ⟨denominator_positive program _ _ lo, denominator_positive program _ _ hi⟩).returns

theorem count_between.«Supported» (d lowerNum lowerDen upperNum upperDen : Int)
    (lo : denominatorCondition program "lower_den" lowerDen)
    (hi : denominatorCondition program "upper_den" upperDen) :
    (Calls.function program "count_between"
      (values d ⟨lowerNum, lowerDen, upperNum, upperDen⟩)).Supported :=
  (count_between.«Returns» d lowerNum lowerDen upperNum upperDen lo hi).supported

namespace solve

theorem «Returns» (limit lowerNum lowerDen upperNum upperDen : Int)
    (lo : denominatorCondition program "lower_den" lowerDen)
    (hi : denominatorCondition program "upper_den" upperDen) :
    (Calls.function program "solve"
      (values limit ⟨lowerNum, lowerDen, upperNum, upperDen⟩)).Returns :=
  (result limit ⟨lowerNum, lowerDen, upperNum, upperDen⟩
    ⟨denominator_positive program _ _ lo, denominator_positive program _ _ hi⟩).returns

theorem «Equiv(naive)» (limit lowerNum lowerDen upperNum upperDen : Int)
    (lo : denominatorCondition program "lower_den" lowerDen)
    (hi : denominatorCondition program "upper_den" upperDen) :
    (Calls.function program "solve" (values limit ⟨lowerNum, lowerDen, upperNum, upperDen⟩)).Equiv
      (Calls.function reference "naive" (values limit ⟨lowerNum, lowerDen, upperNum, upperDen⟩)) :=
  (result limit ⟨lowerNum, lowerDen, upperNum, upperDen⟩
    ⟨denominator_positive program _ _ lo, denominator_positive program _ _ hi⟩).equiv
    (data.«project-euler-statements».data.py.project_euler_statements.p73.naive.result
      limit ⟨lowerNum, lowerDen, upperNum, upperDen⟩)

theorem «Supported» :
    (∀ limit ln ld un ud : Int, denominatorCondition program "lower_den" ld →
      denominatorCondition program "upper_den" ud →
      (Calls.function program "solve" (values limit ⟨ln, ld, un, ud⟩)).Supported) ∧
    (∀ limit ln ld un ud : Int, denominatorCondition program "lower_den" ld →
      denominatorCondition program "upper_den" ud →
      (Calls.function reference "naive" (values limit ⟨ln, ld, un, ud⟩)).Supported) := by
  constructor
  · intro limit ln ld un ud lo hi
    exact («Returns» limit ln ld un ud lo hi).supported
  · intro limit ln ld un ud _ _
    exact (data.«project-euler-statements».data.py.project_euler_statements.p73.naive.result
      limit ⟨ln, ld, un, ud⟩).returns.supported

theorem defaults_returns (limit : Int) (p : Bounds) (positive : Positive p)
    (supplied : List Constant) (omitted : naiveArgs.length ≠ supplied.length)
    (completed : Calls.completeArguments naiveArgs supplied = some (values limit p)) :
    (Calls.function program "solve" supplied).Returns :=
  (default_result limit p positive supplied omitted completed).returns

theorem defaults_equiv (limit : Int) (p : Bounds) (positive : Positive p)
    (supplied : List Constant) (omitted : naiveArgs.length ≠ supplied.length)
    (completed : Calls.completeArguments naiveArgs supplied = some (values limit p)) :
    (Calls.function program "solve" supplied).Equiv (Calls.function reference "naive" supplied) :=
  (default_result limit p positive supplied omitted completed).equiv
    (data.«project-euler-statements».data.py.project_euler_statements.p73.naive.default_result
      limit p supplied omitted completed)

theorem one_argument_returns (limit : Int) :
    (Calls.function program "solve" [.int limit]).Returns :=
  defaults_returns limit ⟨1, 3, 1, 2⟩
    ⟨(by decide : (0 : Int) < 3), (by decide : (0 : Int) < 2)⟩ [.int limit] (by simp) rfl

theorem one_argument_equiv (limit : Int) :
    (Calls.function program "solve" [.int limit]).Equiv
      (Calls.function reference "naive" [.int limit]) :=
  defaults_equiv limit ⟨1, 3, 1, 2⟩
    ⟨(by decide : (0 : Int) < 3), (by decide : (0 : Int) < 2)⟩ [.int limit] (by simp) rfl

theorem two_argument_returns (limit ln : Int) :
    (Calls.function program "solve" [.int limit, .int ln]).Returns :=
  defaults_returns limit ⟨ln, 3, 1, 2⟩ ⟨(by decide : (0 : Int) < 3), (by decide : (0 : Int) < 2)⟩
    [.int limit, .int ln] (by simp) rfl

theorem two_argument_equiv (limit ln : Int) :
    (Calls.function program "solve" [.int limit, .int ln]).Equiv
      (Calls.function reference "naive" [.int limit, .int ln]) :=
  defaults_equiv limit ⟨ln, 3, 1, 2⟩ ⟨(by decide : (0 : Int) < 3), (by decide : (0 : Int) < 2)⟩
    [.int limit, .int ln] (by simp) rfl

theorem three_argument_returns (limit ln ld : Int) (positive : 0 < ld) :
    (Calls.function program "solve" [.int limit, .int ln, .int ld]).Returns :=
  defaults_returns limit ⟨ln, ld, 1, 2⟩ ⟨positive, (by decide : (0 : Int) < 2)⟩
    [.int limit, .int ln, .int ld] (by simp) rfl

theorem three_argument_equiv (limit ln ld : Int) (positive : 0 < ld) :
    (Calls.function program "solve" [.int limit, .int ln, .int ld]).Equiv
      (Calls.function reference "naive" [.int limit, .int ln, .int ld]) :=
  defaults_equiv limit ⟨ln, ld, 1, 2⟩ ⟨positive, (by decide : (0 : Int) < 2)⟩
    [.int limit, .int ln, .int ld] (by simp) rfl

theorem four_argument_returns (limit ln ld un : Int) (positive : 0 < ld) :
    (Calls.function program "solve" [.int limit, .int ln, .int ld, .int un]).Returns :=
  defaults_returns limit ⟨ln, ld, un, 2⟩ ⟨positive, (by decide : (0 : Int) < 2)⟩
    [.int limit, .int ln, .int ld, .int un] (by simp) rfl

theorem four_argument_equiv (limit ln ld un : Int) (positive : 0 < ld) :
    (Calls.function program "solve" [.int limit, .int ln, .int ld, .int un]).Equiv
      (Calls.function reference "naive" [.int limit, .int ln, .int ld, .int un]) :=
  defaults_equiv limit ⟨ln, ld, un, 2⟩ ⟨positive, (by decide : (0 : Int) < 2)⟩
    [.int limit, .int ln, .int ld, .int un] (by simp) rfl

end solve
end solvers.«73»
