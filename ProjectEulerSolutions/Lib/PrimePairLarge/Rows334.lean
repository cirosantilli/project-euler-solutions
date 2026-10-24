import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row11057_small : List Nat := []
theorem row11057_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11057 row11057_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11057_block_005 : List Nat := []
theorem row11057_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11057 row11057_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11057_block_006 : List Nat := [11351, 11939, 11969, 12071, 12197]
def plan11057_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 31), .first, .first, (.firstFactor 13), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .second, .first, .first, (.secondFactor 7), .first, (.secondFactor 17), (.secondFactor 29), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 19), (.secondFactor 13), (.firstFactor 17), .first, .first, .first, .possible, .first, .first, .first, .possible, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .possible, .first, .first, (.firstFactor 13), (.firstFactor 37), .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .possible, (.firstFactor 29), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row11057_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11057 row11057_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11057_block_006
  · decide
  · rfl

def row11057_block_007 : List Nat := [13241, 14369]
def plan11057_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, .second, .first, .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), (.secondFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .second, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 23), (.secondFactor 37), (.secondFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 31), .first, .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 13), .possible, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 29), .first, .first, .second, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 11), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .second, .first, .first, (.secondFactor 17), (.secondFactor 13), .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .second, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 31), .first, .first, .first, (.firstFactor 37), .second, .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 37), .first, .first, .possible, (.secondFactor 7), .first, .first, .first, .second, .first, (.firstFactor 11), .first, .first, .second, .first, .first]
set_option maxRecDepth 2048 in
theorem row11057_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11057 row11057_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11057_block_007
  · decide
  · rfl

def row11057_block_008 : List Nat := [14669, 14741, 14759]
def plan11057_block_008 : List Budget.Choice := [.first, (.firstFactor 11), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 11), (.firstFactor 23), .first, (.firstFactor 19), (.firstFactor 29), .first, .first, .possible, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .possible, (.firstFactor 19), (.firstFactor 11), .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .second, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 17), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11057_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11057 row11057_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11057_block_008
  · decide
  · rfl

def row11057_block_009 : List Nat := []
theorem row11057_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11057 row11057_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11057_block_010 : List Nat := []
theorem row11057_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11057 row11057_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11057_block_011 : List Nat := []
theorem row11057_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11057 row11057_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11057_block_012 : List Nat := []
theorem row11057_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11057 row11057_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11057_block_013 : List Nat := []
theorem row11057_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11057 row11057_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11057_large_left_left : List Nat := row11057_small ++ row11057_block_005
theorem row11057_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11057 row11057_large_left_left :=
  Budget.RowUpperBound.branch row11057_small_covers row11057_block_005_covers

def row11057_large_left_right_right : List Nat := row11057_block_007 ++ row11057_block_008
theorem row11057_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11057 row11057_large_left_right_right :=
  Budget.RowUpperBound.branch row11057_block_007_covers row11057_block_008_covers

def row11057_large_left_right : List Nat := row11057_block_006 ++ row11057_large_left_right_right
theorem row11057_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11057 row11057_large_left_right :=
  Budget.RowUpperBound.branch row11057_block_006_covers row11057_large_left_right_right_covers

def row11057_large_left : List Nat := row11057_large_left_left ++ row11057_large_left_right
theorem row11057_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11057 row11057_large_left :=
  Budget.RowUpperBound.branch row11057_large_left_left_covers row11057_large_left_right_covers

def row11057_large_right_left : List Nat := row11057_block_009 ++ row11057_block_010
theorem row11057_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11057 row11057_large_right_left :=
  Budget.RowUpperBound.branch row11057_block_009_covers row11057_block_010_covers

def row11057_large_right_right_right : List Nat := row11057_block_012 ++ row11057_block_013
theorem row11057_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11057 row11057_large_right_right_right :=
  Budget.RowUpperBound.branch row11057_block_012_covers row11057_block_013_covers

def row11057_large_right_right : List Nat := row11057_block_011 ++ row11057_large_right_right_right
theorem row11057_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11057 row11057_large_right_right :=
  Budget.RowUpperBound.branch row11057_block_011_covers row11057_large_right_right_right_covers

def row11057_large_right : List Nat := row11057_large_right_left ++ row11057_large_right_right
theorem row11057_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11057 row11057_large_right :=
  Budget.RowUpperBound.branch row11057_large_right_left_covers row11057_large_right_right_covers

def row11057_large : List Nat := row11057_large_left ++ row11057_large_right
theorem row11057_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11057 row11057_large :=
  Budget.RowUpperBound.branch row11057_large_left_covers row11057_large_right_covers

def row11057 : List Nat := row11057_large
theorem row11057_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11057 row11057 := row11057_large_covers

def row11059_small : List Nat := []
theorem row11059_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11059 row11059_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11059_block_005 : List Nat := []
theorem row11059_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11059 row11059_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11059_block_006 : List Nat := [11239, 11437, 11731, 12163]
def plan11059_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .second, (.secondFactor 19), .first, .first, (.secondFactor 29), .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 19), .first, .possible, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .second, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .possible, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 29), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .possible, .second, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .second, .first, .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 19), (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 23), .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, .possible, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 37), .first, (.firstFactor 23), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row11059_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11059 row11059_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11059_block_006
  · decide
  · rfl

def row11059_block_007 : List Nat := [12739, 13789]
def plan11059_block_007 : List Budget.Choice := [(.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 17), .first, (.firstFactor 7), (.secondFactor 37), (.secondFactor 7), .second, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .possible, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 29), .second, .first, .first, (.firstFactor 11), .first, .first, .second, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 23), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 7), (.firstFactor 19), .second, .first, .second, .first, .first, (.secondFactor 13), .first, .second, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 19), (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 37), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 23), .first, (.firstFactor 17), (.secondFactor 7), .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 23), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .second, (.firstFactor 13), .first, (.secondFactor 29), .first, (.secondFactor 13), (.firstFactor 19), .first, (.firstFactor 29), .first]
set_option maxRecDepth 2048 in
theorem row11059_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11059 row11059_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11059_block_007
  · decide
  · rfl

def row11059_block_008 : List Nat := []
def plan11059_block_008 : List Budget.Choice := [(.secondFactor 7), .first, (.secondFactor 19), .first, .second, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 19), .first, (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 31), .first, .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11059_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11059 row11059_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11059_block_008
  · decide
  · rfl

def row11059_block_009 : List Nat := []
theorem row11059_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11059 row11059_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11059_block_010 : List Nat := []
theorem row11059_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11059 row11059_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11059_block_011 : List Nat := []
theorem row11059_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11059 row11059_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11059_block_012 : List Nat := []
theorem row11059_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11059 row11059_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11059_block_013 : List Nat := []
theorem row11059_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11059 row11059_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11059_large_left_left : List Nat := row11059_small ++ row11059_block_005
theorem row11059_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11059 row11059_large_left_left :=
  Budget.RowUpperBound.branch row11059_small_covers row11059_block_005_covers

def row11059_large_left_right_right : List Nat := row11059_block_007 ++ row11059_block_008
theorem row11059_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11059 row11059_large_left_right_right :=
  Budget.RowUpperBound.branch row11059_block_007_covers row11059_block_008_covers

def row11059_large_left_right : List Nat := row11059_block_006 ++ row11059_large_left_right_right
theorem row11059_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11059 row11059_large_left_right :=
  Budget.RowUpperBound.branch row11059_block_006_covers row11059_large_left_right_right_covers

def row11059_large_left : List Nat := row11059_large_left_left ++ row11059_large_left_right
theorem row11059_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11059 row11059_large_left :=
  Budget.RowUpperBound.branch row11059_large_left_left_covers row11059_large_left_right_covers

def row11059_large_right_left : List Nat := row11059_block_009 ++ row11059_block_010
theorem row11059_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11059 row11059_large_right_left :=
  Budget.RowUpperBound.branch row11059_block_009_covers row11059_block_010_covers

def row11059_large_right_right_right : List Nat := row11059_block_012 ++ row11059_block_013
theorem row11059_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11059 row11059_large_right_right_right :=
  Budget.RowUpperBound.branch row11059_block_012_covers row11059_block_013_covers

def row11059_large_right_right : List Nat := row11059_block_011 ++ row11059_large_right_right_right
theorem row11059_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11059 row11059_large_right_right :=
  Budget.RowUpperBound.branch row11059_block_011_covers row11059_large_right_right_right_covers

def row11059_large_right : List Nat := row11059_large_right_left ++ row11059_large_right_right
theorem row11059_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11059 row11059_large_right :=
  Budget.RowUpperBound.branch row11059_large_right_left_covers row11059_large_right_right_covers

def row11059_large : List Nat := row11059_large_left ++ row11059_large_right
theorem row11059_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11059 row11059_large :=
  Budget.RowUpperBound.branch row11059_large_left_covers row11059_large_right_covers

def row11059 : List Nat := row11059_large
theorem row11059_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11059 row11059 := row11059_large_covers

def row11069_small : List Nat := []
theorem row11069_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11069 row11069_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11069_block_005 : List Nat := []
theorem row11069_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11069 row11069_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11069_block_006 : List Nat := [11171, 11279, 12269]
def plan11069_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .possible, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 29), .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 13), (.secondFactor 7), (.secondFactor 31), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .second, .first, (.firstFactor 11), (.firstFactor 29), (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 7), (.secondFactor 31), .first, .second, .first, (.firstFactor 17), .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 37), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 37), .first, (.secondFactor 7), .first, .first, .possible, .first, (.firstFactor 29), .first, .first, (.firstFactor 11), (.firstFactor 23), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .second]
set_option maxRecDepth 2048 in
theorem row11069_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11069 row11069_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11069_block_006
  · decide
  · rfl

def row11069_block_007 : List Nat := [12689, 12941, 14153, 14423]
def plan11069_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, (.secondFactor 19), (.secondFactor 23), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 7), .possible, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 7), .possible, (.secondFactor 13), (.firstFactor 17), .first, .first, .first, (.firstFactor 11), (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 19), (.firstFactor 11), .first, .first, .first, (.secondFactor 31), (.secondFactor 13), .first, (.firstFactor 37), .first, (.firstFactor 29), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 31), (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 17), .first, .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 23), .first, (.firstFactor 17), .first, (.secondFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 31), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, .possible, (.firstFactor 13), .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .second, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, (.secondFactor 19), .first, .possible, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row11069_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11069 row11069_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11069_block_007
  · decide
  · rfl

def row11069_block_008 : List Nat := []
def plan11069_block_008 : List Budget.Choice := [.first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 23), (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 19), .first, (.secondFactor 29), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .second, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .second, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11069_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11069 row11069_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11069_block_008
  · decide
  · rfl

def row11069_block_009 : List Nat := []
theorem row11069_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11069 row11069_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11069_block_010 : List Nat := []
theorem row11069_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11069 row11069_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11069_block_011 : List Nat := []
theorem row11069_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11069 row11069_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11069_block_012 : List Nat := []
theorem row11069_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11069 row11069_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11069_block_013 : List Nat := []
theorem row11069_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11069 row11069_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11069_large_left_left : List Nat := row11069_small ++ row11069_block_005
theorem row11069_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11069 row11069_large_left_left :=
  Budget.RowUpperBound.branch row11069_small_covers row11069_block_005_covers

def row11069_large_left_right_right : List Nat := row11069_block_007 ++ row11069_block_008
theorem row11069_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11069 row11069_large_left_right_right :=
  Budget.RowUpperBound.branch row11069_block_007_covers row11069_block_008_covers

def row11069_large_left_right : List Nat := row11069_block_006 ++ row11069_large_left_right_right
theorem row11069_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11069 row11069_large_left_right :=
  Budget.RowUpperBound.branch row11069_block_006_covers row11069_large_left_right_right_covers

def row11069_large_left : List Nat := row11069_large_left_left ++ row11069_large_left_right
theorem row11069_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11069 row11069_large_left :=
  Budget.RowUpperBound.branch row11069_large_left_left_covers row11069_large_left_right_covers

def row11069_large_right_left : List Nat := row11069_block_009 ++ row11069_block_010
theorem row11069_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11069 row11069_large_right_left :=
  Budget.RowUpperBound.branch row11069_block_009_covers row11069_block_010_covers

def row11069_large_right_right_right : List Nat := row11069_block_012 ++ row11069_block_013
theorem row11069_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11069 row11069_large_right_right_right :=
  Budget.RowUpperBound.branch row11069_block_012_covers row11069_block_013_covers

def row11069_large_right_right : List Nat := row11069_block_011 ++ row11069_large_right_right_right
theorem row11069_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11069 row11069_large_right_right :=
  Budget.RowUpperBound.branch row11069_block_011_covers row11069_large_right_right_right_covers

def row11069_large_right : List Nat := row11069_large_right_left ++ row11069_large_right_right
theorem row11069_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11069 row11069_large_right :=
  Budget.RowUpperBound.branch row11069_large_right_left_covers row11069_large_right_right_covers

def row11069_large : List Nat := row11069_large_left ++ row11069_large_right
theorem row11069_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11069 row11069_large :=
  Budget.RowUpperBound.branch row11069_large_left_covers row11069_large_right_covers

def row11069 : List Nat := row11069_large
theorem row11069_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11069 row11069 := row11069_large_covers

def row11071_small : List Nat := []
theorem row11071_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11071 row11071_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11071_block_005 : List Nat := []
theorem row11071_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11071 row11071_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11071_block_006 : List Nat := [11443, 11491, 12073]
def plan11071_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 13), .first, .first, (.secondFactor 13), .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 29), .first, (.secondFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .possible, .first, (.firstFactor 7), .first, .first, .first, .possible, (.secondFactor 17), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 29), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, .second, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 13), (.secondFactor 19), (.firstFactor 31), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, .possible, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 31), .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 37), .first, (.firstFactor 13), .first, .first, .second, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first]
set_option maxRecDepth 2048 in
theorem row11071_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11071 row11071_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11071_block_006
  · decide
  · rfl

def row11071_block_007 : List Nat := [13291, 14293]
def plan11071_block_007 : List Budget.Choice := [.first, (.firstFactor 7), .first, .second, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 29), (.secondFactor 7), .first, (.firstFactor 19), (.secondFactor 23), (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 11), .second, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 17), (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .second, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 19), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .possible, (.firstFactor 37), (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 23), (.secondFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 17), (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 17), (.firstFactor 37), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 7), .possible, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 19), .first, .first, (.secondFactor 13), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 19)]
set_option maxRecDepth 2048 in
theorem row11071_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11071 row11071_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11071_block_007
  · decide
  · rfl

def row11071_block_008 : List Nat := []
def plan11071_block_008 : List Budget.Choice := [.first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 31), .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 7), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11071_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11071 row11071_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11071_block_008
  · decide
  · rfl

def row11071_block_009 : List Nat := []
theorem row11071_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11071 row11071_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11071_block_010 : List Nat := []
theorem row11071_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11071 row11071_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11071_block_011 : List Nat := []
theorem row11071_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11071 row11071_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11071_block_012 : List Nat := []
theorem row11071_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11071 row11071_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11071_block_013 : List Nat := []
theorem row11071_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11071 row11071_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11071_large_left_left : List Nat := row11071_small ++ row11071_block_005
theorem row11071_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11071 row11071_large_left_left :=
  Budget.RowUpperBound.branch row11071_small_covers row11071_block_005_covers

def row11071_large_left_right_right : List Nat := row11071_block_007 ++ row11071_block_008
theorem row11071_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11071 row11071_large_left_right_right :=
  Budget.RowUpperBound.branch row11071_block_007_covers row11071_block_008_covers

def row11071_large_left_right : List Nat := row11071_block_006 ++ row11071_large_left_right_right
theorem row11071_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11071 row11071_large_left_right :=
  Budget.RowUpperBound.branch row11071_block_006_covers row11071_large_left_right_right_covers

def row11071_large_left : List Nat := row11071_large_left_left ++ row11071_large_left_right
theorem row11071_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11071 row11071_large_left :=
  Budget.RowUpperBound.branch row11071_large_left_left_covers row11071_large_left_right_covers

def row11071_large_right_left : List Nat := row11071_block_009 ++ row11071_block_010
theorem row11071_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11071 row11071_large_right_left :=
  Budget.RowUpperBound.branch row11071_block_009_covers row11071_block_010_covers

def row11071_large_right_right_right : List Nat := row11071_block_012 ++ row11071_block_013
theorem row11071_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11071 row11071_large_right_right_right :=
  Budget.RowUpperBound.branch row11071_block_012_covers row11071_block_013_covers

def row11071_large_right_right : List Nat := row11071_block_011 ++ row11071_large_right_right_right
theorem row11071_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11071 row11071_large_right_right :=
  Budget.RowUpperBound.branch row11071_block_011_covers row11071_large_right_right_right_covers

def row11071_large_right : List Nat := row11071_large_right_left ++ row11071_large_right_right
theorem row11071_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11071 row11071_large_right :=
  Budget.RowUpperBound.branch row11071_large_right_left_covers row11071_large_right_right_covers

def row11071_large : List Nat := row11071_large_left ++ row11071_large_right
theorem row11071_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11071 row11071_large :=
  Budget.RowUpperBound.branch row11071_large_left_covers row11071_large_right_covers

def row11071 : List Nat := row11071_large
theorem row11071_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11071 row11071 := row11071_large_covers

end ProjectEulerSolutions.PrimePairLarge
