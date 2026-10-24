import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row10301_small : List Nat := []
theorem row10301_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10301 row10301_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10301_block_005 : List Nat := [10343]
def plan10301_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .possible, .first, .first]
set_option maxRecDepth 2048 in
theorem row10301_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10301 row10301_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10301_block_005
  · decide
  · rfl

def row10301_block_006 : List Nat := [10889, 11717, 11933]
def plan10301_block_006 : List Budget.Choice := [.first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .second, (.firstFactor 19), .first, .first, (.secondFactor 13), .possible, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 19), .second, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 19), .first, (.firstFactor 13), .first, .first, (.firstFactor 19), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 7), .first, .first, .second, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 17), (.secondFactor 19), (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 13), .possible, .second, .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 29), .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 19), (.secondFactor 7), (.secondFactor 29), (.firstFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 19), (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row10301_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10301 row10301_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10301_block_006
  · decide
  · rfl

def row10301_block_007 : List Nat := [12527, 12641, 12899, 13697]
def plan10301_block_007 : List Budget.Choice := [.first, .first, .second, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 29), (.secondFactor 7), (.firstFactor 7), .first, .first, .possible, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .possible, .first, (.secondFactor 19), .first, (.firstFactor 7), (.secondFactor 37), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 37), .possible, .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 31), (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 23), (.secondFactor 7), .first, .first, .first, .second, .first, .first, .first, (.secondFactor 29), (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 19), (.secondFactor 13), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), .second, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 37), .first, .first, (.secondFactor 13), .second, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .possible, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .second, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 37), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .second, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 29), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 31), (.firstFactor 31), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 37), (.firstFactor 29), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 37), .first, .first]
set_option maxRecDepth 2048 in
theorem row10301_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10301 row10301_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10301_block_007
  · decide
  · rfl

def row10301_block_008 : List Nat := [14867]
def plan10301_block_008 : List Budget.Choice := [.first, .first, .first, (.firstFactor 7), .first, .first, .second, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 19), (.secondFactor 13), .first, (.secondFactor 23), (.firstFactor 23), .first, (.firstFactor 11), (.firstFactor 37), .first, .first, .first, (.firstFactor 31), (.firstFactor 11), .first, .first, .second, .first, (.secondFactor 29), (.secondFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .possible, .first, .second, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 19), (.secondFactor 31), .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 37), .second, .first, (.firstFactor 7), (.secondFactor 19), (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 23), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 29), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10301_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10301 row10301_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10301_block_008
  · decide
  · rfl

def row10301_block_009 : List Nat := []
theorem row10301_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10301 row10301_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10301_block_010 : List Nat := []
theorem row10301_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10301 row10301_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10301_block_011 : List Nat := []
theorem row10301_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10301 row10301_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10301_block_012 : List Nat := []
theorem row10301_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10301 row10301_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10301_block_013 : List Nat := []
theorem row10301_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10301 row10301_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10301_large_left_left : List Nat := row10301_small ++ row10301_block_005
theorem row10301_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10301 row10301_large_left_left :=
  Budget.RowUpperBound.branch row10301_small_covers row10301_block_005_covers

def row10301_large_left_right_right : List Nat := row10301_block_007 ++ row10301_block_008
theorem row10301_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10301 row10301_large_left_right_right :=
  Budget.RowUpperBound.branch row10301_block_007_covers row10301_block_008_covers

def row10301_large_left_right : List Nat := row10301_block_006 ++ row10301_large_left_right_right
theorem row10301_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10301 row10301_large_left_right :=
  Budget.RowUpperBound.branch row10301_block_006_covers row10301_large_left_right_right_covers

def row10301_large_left : List Nat := row10301_large_left_left ++ row10301_large_left_right
theorem row10301_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10301 row10301_large_left :=
  Budget.RowUpperBound.branch row10301_large_left_left_covers row10301_large_left_right_covers

def row10301_large_right_left : List Nat := row10301_block_009 ++ row10301_block_010
theorem row10301_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10301 row10301_large_right_left :=
  Budget.RowUpperBound.branch row10301_block_009_covers row10301_block_010_covers

def row10301_large_right_right_right : List Nat := row10301_block_012 ++ row10301_block_013
theorem row10301_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10301 row10301_large_right_right_right :=
  Budget.RowUpperBound.branch row10301_block_012_covers row10301_block_013_covers

def row10301_large_right_right : List Nat := row10301_block_011 ++ row10301_large_right_right_right
theorem row10301_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10301 row10301_large_right_right :=
  Budget.RowUpperBound.branch row10301_block_011_covers row10301_large_right_right_right_covers

def row10301_large_right : List Nat := row10301_large_right_left ++ row10301_large_right_right
theorem row10301_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10301 row10301_large_right :=
  Budget.RowUpperBound.branch row10301_large_right_left_covers row10301_large_right_right_covers

def row10301_large : List Nat := row10301_large_left ++ row10301_large_right
theorem row10301_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10301 row10301_large :=
  Budget.RowUpperBound.branch row10301_large_left_covers row10301_large_right_covers

def row10301 : List Nat := row10301_large
theorem row10301_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10301 row10301 := row10301_large_covers

def row10303_small : List Nat := []
theorem row10303_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10303 row10303_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10303_block_005 : List Nat := []
def plan10303_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row10303_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10303 row10303_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10303_block_005
  · decide
  · rfl

def row10303_block_006 : List Nat := [10861, 11083, 11131, 11551, 11719, 11941]
def plan10303_block_006 : List Budget.Choice := [.first, (.firstFactor 37), .first, .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 31), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.secondFactor 17), (.secondFactor 19), (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 11), (.secondFactor 13), .first, .first, .first, .possible, (.secondFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 29), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .second, (.firstFactor 17), .first, .first, (.secondFactor 17), .first, (.firstFactor 23), .first, (.firstFactor 31), .possible, .first, .first, (.firstFactor 13), .first, (.firstFactor 19), .possible, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 17), (.firstFactor 31), .first, .first, .first, (.firstFactor 37), (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .possible, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 23), .first, .possible, (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .second, .first, (.secondFactor 7), (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .possible, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 31), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .second, .first, .second, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 7), (.secondFactor 23), .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row10303_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10303 row10303_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10303_block_006
  · decide
  · rfl

def row10303_block_007 : List Nat := [12721, 13327, 13759, 13831]
def plan10303_block_007 : List Budget.Choice := [(.firstFactor 17), .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .possible, .first, .first, (.secondFactor 7), .second, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .second, (.firstFactor 7), (.firstFactor 29), (.firstFactor 23), .first, .first, .first, (.firstFactor 13), .first, .first, .second, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 7), (.firstFactor 13), .first, .first, .possible, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 37), .first, .first, .possible, .first, .first, .first, .first, (.secondFactor 7), .first, .possible, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 19), .first, (.secondFactor 17), (.secondFactor 7), (.secondFactor 37), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .second, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 13), .first, (.firstFactor 29), .first, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row10303_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10303 row10303_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10303_block_007
  · decide
  · rfl

def row10303_block_008 : List Nat := [14533, 14851]
def plan10303_block_008 : List Budget.Choice := [(.secondFactor 7), .first, (.secondFactor 13), .first, .possible, .first, .first, .first, (.firstFactor 7), (.secondFactor 29), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .second, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .second, .first, (.secondFactor 17), .first, .first, .first, .possible, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .second, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .second, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), (.secondFactor 13), .first, (.firstFactor 11), .first, .first, .first, .second, .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .second, (.secondFactor 19), .first, (.firstFactor 19), .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10303_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10303 row10303_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10303_block_008
  · decide
  · rfl

def row10303_block_009 : List Nat := []
theorem row10303_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10303 row10303_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10303_block_010 : List Nat := []
theorem row10303_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10303 row10303_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10303_block_011 : List Nat := []
theorem row10303_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10303 row10303_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10303_block_012 : List Nat := []
theorem row10303_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10303 row10303_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10303_block_013 : List Nat := []
theorem row10303_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10303 row10303_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10303_large_left_left : List Nat := row10303_small ++ row10303_block_005
theorem row10303_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10303 row10303_large_left_left :=
  Budget.RowUpperBound.branch row10303_small_covers row10303_block_005_covers

def row10303_large_left_right_right : List Nat := row10303_block_007 ++ row10303_block_008
theorem row10303_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10303 row10303_large_left_right_right :=
  Budget.RowUpperBound.branch row10303_block_007_covers row10303_block_008_covers

def row10303_large_left_right : List Nat := row10303_block_006 ++ row10303_large_left_right_right
theorem row10303_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10303 row10303_large_left_right :=
  Budget.RowUpperBound.branch row10303_block_006_covers row10303_large_left_right_right_covers

def row10303_large_left : List Nat := row10303_large_left_left ++ row10303_large_left_right
theorem row10303_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10303 row10303_large_left :=
  Budget.RowUpperBound.branch row10303_large_left_left_covers row10303_large_left_right_covers

def row10303_large_right_left : List Nat := row10303_block_009 ++ row10303_block_010
theorem row10303_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10303 row10303_large_right_left :=
  Budget.RowUpperBound.branch row10303_block_009_covers row10303_block_010_covers

def row10303_large_right_right_right : List Nat := row10303_block_012 ++ row10303_block_013
theorem row10303_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10303 row10303_large_right_right_right :=
  Budget.RowUpperBound.branch row10303_block_012_covers row10303_block_013_covers

def row10303_large_right_right : List Nat := row10303_block_011 ++ row10303_large_right_right_right
theorem row10303_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10303 row10303_large_right_right :=
  Budget.RowUpperBound.branch row10303_block_011_covers row10303_large_right_right_right_covers

def row10303_large_right : List Nat := row10303_large_right_left ++ row10303_large_right_right
theorem row10303_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10303 row10303_large_right :=
  Budget.RowUpperBound.branch row10303_large_right_left_covers row10303_large_right_right_covers

def row10303_large : List Nat := row10303_large_left ++ row10303_large_right
theorem row10303_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10303 row10303_large :=
  Budget.RowUpperBound.branch row10303_large_left_covers row10303_large_right_covers

def row10303 : List Nat := row10303_large
theorem row10303_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10303 row10303 := row10303_large_covers

def row10313_small : List Nat := []
theorem row10313_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10313 row10313_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10313_block_005 : List Nat := []
def plan10313_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 19), .first, (.secondFactor 17), (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row10313_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10313 row10313_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10313_block_005
  · decide
  · rfl

def row10313_block_006 : List Nat := [10391, 10799, 11399, 11657]
def plan10313_block_006 : List Budget.Choice := [.possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), (.firstFactor 29), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .second, .first, .second, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .second, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .possible, .first, .first, (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .second, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .possible, (.firstFactor 13), .first, .first, .first, .second, .first, (.firstFactor 37), .second, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .second, .first, (.firstFactor 19), (.firstFactor 11), .possible, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 17), (.firstFactor 23), (.firstFactor 13), (.secondFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), (.secondFactor 37), .first, .first, (.firstFactor 23), (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 23), .first, (.firstFactor 11), (.firstFactor 29), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .second, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 29)]
set_option maxRecDepth 2048 in
theorem row10313_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10313 row10313_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10313_block_006
  · decide
  · rfl

def row10313_block_007 : List Nat := [12437, 13757, 14087]
def plan10313_block_007 : List Budget.Choice := [.first, .first, .possible, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), (.firstFactor 17), (.firstFactor 13), .first, .first, (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 19), (.firstFactor 7), (.firstFactor 13), (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 13), .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 23), .second, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .second, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 19), (.firstFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 31), .first, .first, .first, (.secondFactor 23), (.firstFactor 11), .first, (.secondFactor 29), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 19), (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), (.secondFactor 23), (.firstFactor 37), .first, (.firstFactor 31), (.secondFactor 17), .first, (.secondFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .possible, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 23), .first, (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 19), .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .second, (.secondFactor 7), .first, (.firstFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 29), .first, .first, .first, .second, .first, .first, .first, (.firstFactor 23), .first, .second, .first, .first, (.secondFactor 29), .first, .first]
set_option maxRecDepth 2048 in
theorem row10313_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10313 row10313_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10313_block_007
  · decide
  · rfl

def row10313_block_008 : List Nat := [14759, 15083, 15629]
def plan10313_block_008 : List Budget.Choice := [.first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .second, (.secondFactor 17), .first, .first, (.firstFactor 17), .second, (.firstFactor 7), .possible, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 19), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 29), .first, .first, .second, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .possible, .first, (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 19), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .second, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 31), .first, .first, .first, .first, .possible, (.secondFactor 17), .first, .first, .first, .first, .first, (.secondFactor 23), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10313_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10313 row10313_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10313_block_008
  · decide
  · rfl

def row10313_block_009 : List Nat := []
theorem row10313_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10313 row10313_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10313_block_010 : List Nat := []
theorem row10313_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10313 row10313_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10313_block_011 : List Nat := []
theorem row10313_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10313 row10313_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10313_block_012 : List Nat := []
theorem row10313_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10313 row10313_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10313_block_013 : List Nat := []
theorem row10313_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10313 row10313_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10313_large_left_left : List Nat := row10313_small ++ row10313_block_005
theorem row10313_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10313 row10313_large_left_left :=
  Budget.RowUpperBound.branch row10313_small_covers row10313_block_005_covers

def row10313_large_left_right_right : List Nat := row10313_block_007 ++ row10313_block_008
theorem row10313_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10313 row10313_large_left_right_right :=
  Budget.RowUpperBound.branch row10313_block_007_covers row10313_block_008_covers

def row10313_large_left_right : List Nat := row10313_block_006 ++ row10313_large_left_right_right
theorem row10313_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10313 row10313_large_left_right :=
  Budget.RowUpperBound.branch row10313_block_006_covers row10313_large_left_right_right_covers

def row10313_large_left : List Nat := row10313_large_left_left ++ row10313_large_left_right
theorem row10313_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10313 row10313_large_left :=
  Budget.RowUpperBound.branch row10313_large_left_left_covers row10313_large_left_right_covers

def row10313_large_right_left : List Nat := row10313_block_009 ++ row10313_block_010
theorem row10313_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10313 row10313_large_right_left :=
  Budget.RowUpperBound.branch row10313_block_009_covers row10313_block_010_covers

def row10313_large_right_right_right : List Nat := row10313_block_012 ++ row10313_block_013
theorem row10313_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10313 row10313_large_right_right_right :=
  Budget.RowUpperBound.branch row10313_block_012_covers row10313_block_013_covers

def row10313_large_right_right : List Nat := row10313_block_011 ++ row10313_large_right_right_right
theorem row10313_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10313 row10313_large_right_right :=
  Budget.RowUpperBound.branch row10313_block_011_covers row10313_large_right_right_right_covers

def row10313_large_right : List Nat := row10313_large_right_left ++ row10313_large_right_right
theorem row10313_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10313 row10313_large_right :=
  Budget.RowUpperBound.branch row10313_large_right_left_covers row10313_large_right_right_covers

def row10313_large : List Nat := row10313_large_left ++ row10313_large_right
theorem row10313_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10313 row10313_large :=
  Budget.RowUpperBound.branch row10313_large_left_covers row10313_large_right_covers

def row10313 : List Nat := row10313_large
theorem row10313_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10313 row10313 := row10313_large_covers

def row10321_small : List Nat := []
theorem row10321_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10321 row10321_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10321_block_005 : List Nat := []
def plan10321_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 31), .first, .first, (.secondFactor 29), .first]
set_option maxRecDepth 2048 in
theorem row10321_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10321 row10321_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10321_block_005
  · decide
  · rfl

def row10321_block_006 : List Nat := [11287]
def plan10321_block_006 : List Budget.Choice := [.first, .second, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 19), .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .second, (.firstFactor 19), .first, .first, .first, (.secondFactor 37), .first, .first, .first, (.firstFactor 7), .second, .second, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 17), (.firstFactor 23), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 37), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 23), (.secondFactor 7), .first, .first, .first, .second, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, .possible, (.secondFactor 17), (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .second, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 13), (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 29), .first, .first, (.firstFactor 11), (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 31), .first, .first, (.secondFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row10321_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10321 row10321_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10321_block_006
  · decide
  · rfl

def row10321_block_007 : List Nat := [12457, 12973, 13513, 14107]
def plan10321_block_007 : List Budget.Choice := [(.secondFactor 17), (.firstFactor 11), .first, (.secondFactor 7), .possible, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 17), .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), (.secondFactor 19), .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 13), (.firstFactor 31), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 29), .possible, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 7), .first, (.secondFactor 19), (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 31), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .possible, (.firstFactor 19), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 31), (.firstFactor 11), .first, .first, .first, (.secondFactor 13), (.secondFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 23), (.firstFactor 29), .first, .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row10321_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10321 row10321_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10321_block_007
  · decide
  · rfl

def row10321_block_008 : List Nat := [15187, 15241, 15619]
def plan10321_block_008 : List Budget.Choice := [(.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 13), (.firstFactor 17), .first, (.secondFactor 17), .first, (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .second, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 23), (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 11), .first, .first, .first, .possible, .first, .first, (.firstFactor 7), .first, .first, .possible, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 17), .first, .first, .first, (.firstFactor 29), .second, (.secondFactor 19), .first, (.firstFactor 13), (.secondFactor 7), .first, .second, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 31), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 11), .first, .possible, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), (.secondFactor 19), (.firstFactor 11), .first, (.firstFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10321_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10321 row10321_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10321_block_008
  · decide
  · rfl

def row10321_block_009 : List Nat := []
theorem row10321_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10321 row10321_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10321_block_010 : List Nat := []
theorem row10321_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10321 row10321_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10321_block_011 : List Nat := []
theorem row10321_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10321 row10321_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10321_block_012 : List Nat := []
theorem row10321_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10321 row10321_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10321_block_013 : List Nat := []
theorem row10321_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10321 row10321_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10321_large_left_left : List Nat := row10321_small ++ row10321_block_005
theorem row10321_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10321 row10321_large_left_left :=
  Budget.RowUpperBound.branch row10321_small_covers row10321_block_005_covers

def row10321_large_left_right_right : List Nat := row10321_block_007 ++ row10321_block_008
theorem row10321_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10321 row10321_large_left_right_right :=
  Budget.RowUpperBound.branch row10321_block_007_covers row10321_block_008_covers

def row10321_large_left_right : List Nat := row10321_block_006 ++ row10321_large_left_right_right
theorem row10321_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10321 row10321_large_left_right :=
  Budget.RowUpperBound.branch row10321_block_006_covers row10321_large_left_right_right_covers

def row10321_large_left : List Nat := row10321_large_left_left ++ row10321_large_left_right
theorem row10321_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10321 row10321_large_left :=
  Budget.RowUpperBound.branch row10321_large_left_left_covers row10321_large_left_right_covers

def row10321_large_right_left : List Nat := row10321_block_009 ++ row10321_block_010
theorem row10321_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10321 row10321_large_right_left :=
  Budget.RowUpperBound.branch row10321_block_009_covers row10321_block_010_covers

def row10321_large_right_right_right : List Nat := row10321_block_012 ++ row10321_block_013
theorem row10321_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10321 row10321_large_right_right_right :=
  Budget.RowUpperBound.branch row10321_block_012_covers row10321_block_013_covers

def row10321_large_right_right : List Nat := row10321_block_011 ++ row10321_large_right_right_right
theorem row10321_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10321 row10321_large_right_right :=
  Budget.RowUpperBound.branch row10321_block_011_covers row10321_large_right_right_right_covers

def row10321_large_right : List Nat := row10321_large_right_left ++ row10321_large_right_right
theorem row10321_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10321 row10321_large_right :=
  Budget.RowUpperBound.branch row10321_large_right_left_covers row10321_large_right_right_covers

def row10321_large : List Nat := row10321_large_left ++ row10321_large_right
theorem row10321_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10321 row10321_large :=
  Budget.RowUpperBound.branch row10321_large_left_covers row10321_large_right_covers

def row10321 : List Nat := row10321_large
theorem row10321_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10321 row10321 := row10321_large_covers

end ProjectEulerSolutions.PrimePairLarge
