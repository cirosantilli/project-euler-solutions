import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row11867_small : List Nat := []
theorem row11867_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11867 row11867_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11867_block_005 : List Nat := []
theorem row11867_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11867 row11867_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11867_block_006 : List Nat := [12113]
def plan11867_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), (.firstFactor 13), .second, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .possible, .first, (.secondFactor 29), (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 11), (.firstFactor 19), .first, (.secondFactor 13), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .second, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row11867_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11867 row11867_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11867_block_006
  · decide
  · rfl

def row11867_block_007 : List Nat := [13907]
def plan11867_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 23), .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 29), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 7), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .second, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 31), .second, .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, (.firstFactor 23), (.secondFactor 19), (.secondFactor 7), .first, .first, .first, .second, .second, (.firstFactor 11), (.secondFactor 23), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 13), (.firstFactor 37), .first, .second, (.secondFactor 19), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.firstFactor 37), .first, (.firstFactor 13), (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .second, .first, (.firstFactor 17), (.firstFactor 29), .first, .possible, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11867_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11867 row11867_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11867_block_007
  · decide
  · rfl

def row11867_block_008 : List Nat := []
theorem row11867_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11867 row11867_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11867_block_009 : List Nat := []
theorem row11867_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11867 row11867_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11867_block_010 : List Nat := []
theorem row11867_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11867 row11867_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11867_block_011 : List Nat := []
theorem row11867_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11867 row11867_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11867_block_012 : List Nat := []
theorem row11867_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11867 row11867_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11867_block_013 : List Nat := []
theorem row11867_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11867 row11867_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11867_large_left_left : List Nat := row11867_small ++ row11867_block_005
theorem row11867_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11867 row11867_large_left_left :=
  Budget.RowUpperBound.branch row11867_small_covers row11867_block_005_covers

def row11867_large_left_right_right : List Nat := row11867_block_007 ++ row11867_block_008
theorem row11867_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11867 row11867_large_left_right_right :=
  Budget.RowUpperBound.branch row11867_block_007_covers row11867_block_008_covers

def row11867_large_left_right : List Nat := row11867_block_006 ++ row11867_large_left_right_right
theorem row11867_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11867 row11867_large_left_right :=
  Budget.RowUpperBound.branch row11867_block_006_covers row11867_large_left_right_right_covers

def row11867_large_left : List Nat := row11867_large_left_left ++ row11867_large_left_right
theorem row11867_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11867 row11867_large_left :=
  Budget.RowUpperBound.branch row11867_large_left_left_covers row11867_large_left_right_covers

def row11867_large_right_left : List Nat := row11867_block_009 ++ row11867_block_010
theorem row11867_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11867 row11867_large_right_left :=
  Budget.RowUpperBound.branch row11867_block_009_covers row11867_block_010_covers

def row11867_large_right_right_right : List Nat := row11867_block_012 ++ row11867_block_013
theorem row11867_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11867 row11867_large_right_right_right :=
  Budget.RowUpperBound.branch row11867_block_012_covers row11867_block_013_covers

def row11867_large_right_right : List Nat := row11867_block_011 ++ row11867_large_right_right_right
theorem row11867_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11867 row11867_large_right_right :=
  Budget.RowUpperBound.branch row11867_block_011_covers row11867_large_right_right_right_covers

def row11867_large_right : List Nat := row11867_large_right_left ++ row11867_large_right_right
theorem row11867_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11867 row11867_large_right :=
  Budget.RowUpperBound.branch row11867_large_right_left_covers row11867_large_right_right_covers

def row11867_large : List Nat := row11867_large_left ++ row11867_large_right
theorem row11867_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11867 row11867_large :=
  Budget.RowUpperBound.branch row11867_large_left_covers row11867_large_right_covers

def row11867 : List Nat := row11867_large
theorem row11867_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11867 row11867 := row11867_large_covers

def row11887_small : List Nat := []
theorem row11887_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11887 row11887_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11887_block_005 : List Nat := []
theorem row11887_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11887 row11887_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11887_block_006 : List Nat := [12043]
def plan11887_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .possible, (.firstFactor 7), .first, (.secondFactor 37), (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .second, (.firstFactor 31), .first, (.firstFactor 19), .first]
set_option maxRecDepth 2048 in
theorem row11887_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11887 row11887_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11887_block_006
  · decide
  · rfl

def row11887_block_007 : List Nat := [14143]
def plan11887_block_007 : List Budget.Choice := [(.firstFactor 13), .first, .first, .second, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 37), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 23), .second, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .second, .first, .first, .first, (.secondFactor 23), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 37), .second, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 31), (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 29), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 11), .first, .second, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, .second, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 17), .first, .first, (.secondFactor 13), .first, (.firstFactor 31), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, (.secondFactor 37), .first, (.firstFactor 23), .first, (.firstFactor 7), .possible, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11887_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11887 row11887_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11887_block_007
  · decide
  · rfl

def row11887_block_008 : List Nat := []
theorem row11887_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11887 row11887_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11887_block_009 : List Nat := []
theorem row11887_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11887 row11887_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11887_block_010 : List Nat := []
theorem row11887_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11887 row11887_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11887_block_011 : List Nat := []
theorem row11887_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11887 row11887_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11887_block_012 : List Nat := []
theorem row11887_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11887 row11887_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11887_block_013 : List Nat := []
theorem row11887_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11887 row11887_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11887_large_left_left : List Nat := row11887_small ++ row11887_block_005
theorem row11887_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11887 row11887_large_left_left :=
  Budget.RowUpperBound.branch row11887_small_covers row11887_block_005_covers

def row11887_large_left_right_right : List Nat := row11887_block_007 ++ row11887_block_008
theorem row11887_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11887 row11887_large_left_right_right :=
  Budget.RowUpperBound.branch row11887_block_007_covers row11887_block_008_covers

def row11887_large_left_right : List Nat := row11887_block_006 ++ row11887_large_left_right_right
theorem row11887_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11887 row11887_large_left_right :=
  Budget.RowUpperBound.branch row11887_block_006_covers row11887_large_left_right_right_covers

def row11887_large_left : List Nat := row11887_large_left_left ++ row11887_large_left_right
theorem row11887_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11887 row11887_large_left :=
  Budget.RowUpperBound.branch row11887_large_left_left_covers row11887_large_left_right_covers

def row11887_large_right_left : List Nat := row11887_block_009 ++ row11887_block_010
theorem row11887_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11887 row11887_large_right_left :=
  Budget.RowUpperBound.branch row11887_block_009_covers row11887_block_010_covers

def row11887_large_right_right_right : List Nat := row11887_block_012 ++ row11887_block_013
theorem row11887_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11887 row11887_large_right_right_right :=
  Budget.RowUpperBound.branch row11887_block_012_covers row11887_block_013_covers

def row11887_large_right_right : List Nat := row11887_block_011 ++ row11887_large_right_right_right
theorem row11887_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11887 row11887_large_right_right :=
  Budget.RowUpperBound.branch row11887_block_011_covers row11887_large_right_right_right_covers

def row11887_large_right : List Nat := row11887_large_right_left ++ row11887_large_right_right
theorem row11887_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11887 row11887_large_right :=
  Budget.RowUpperBound.branch row11887_large_right_left_covers row11887_large_right_right_covers

def row11887_large : List Nat := row11887_large_left ++ row11887_large_right
theorem row11887_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11887 row11887_large :=
  Budget.RowUpperBound.branch row11887_large_left_covers row11887_large_right_covers

def row11887 : List Nat := row11887_large
theorem row11887_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11887 row11887 := row11887_large_covers

def row11897_small : List Nat := []
theorem row11897_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11897 row11897_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11897_block_005 : List Nat := []
theorem row11897_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11897 row11897_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11897_block_006 : List Nat := [12101]
def plan11897_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 29), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .possible, .second, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 19), (.firstFactor 31), .first, (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row11897_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11897 row11897_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11897_block_006
  · decide
  · rfl

def row11897_block_007 : List Nat := [12527, 13781]
def plan11897_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 17), .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .first, .possible, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 19), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 29), .first, .second, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 31), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 19), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .possible, .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 7), (.secondFactor 23), .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 29), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 23), .first, (.firstFactor 11), .first, .first, (.secondFactor 13), (.firstFactor 19), (.firstFactor 7), .first, .second, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11897_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11897 row11897_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11897_block_007
  · decide
  · rfl

def row11897_block_008 : List Nat := []
theorem row11897_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11897 row11897_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11897_block_009 : List Nat := []
theorem row11897_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11897 row11897_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11897_block_010 : List Nat := []
theorem row11897_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11897 row11897_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11897_block_011 : List Nat := []
theorem row11897_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11897 row11897_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11897_block_012 : List Nat := []
theorem row11897_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11897 row11897_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11897_block_013 : List Nat := []
theorem row11897_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11897 row11897_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11897_large_left_left : List Nat := row11897_small ++ row11897_block_005
theorem row11897_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11897 row11897_large_left_left :=
  Budget.RowUpperBound.branch row11897_small_covers row11897_block_005_covers

def row11897_large_left_right_right : List Nat := row11897_block_007 ++ row11897_block_008
theorem row11897_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11897 row11897_large_left_right_right :=
  Budget.RowUpperBound.branch row11897_block_007_covers row11897_block_008_covers

def row11897_large_left_right : List Nat := row11897_block_006 ++ row11897_large_left_right_right
theorem row11897_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11897 row11897_large_left_right :=
  Budget.RowUpperBound.branch row11897_block_006_covers row11897_large_left_right_right_covers

def row11897_large_left : List Nat := row11897_large_left_left ++ row11897_large_left_right
theorem row11897_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11897 row11897_large_left :=
  Budget.RowUpperBound.branch row11897_large_left_left_covers row11897_large_left_right_covers

def row11897_large_right_left : List Nat := row11897_block_009 ++ row11897_block_010
theorem row11897_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11897 row11897_large_right_left :=
  Budget.RowUpperBound.branch row11897_block_009_covers row11897_block_010_covers

def row11897_large_right_right_right : List Nat := row11897_block_012 ++ row11897_block_013
theorem row11897_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11897 row11897_large_right_right_right :=
  Budget.RowUpperBound.branch row11897_block_012_covers row11897_block_013_covers

def row11897_large_right_right : List Nat := row11897_block_011 ++ row11897_large_right_right_right
theorem row11897_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11897 row11897_large_right_right :=
  Budget.RowUpperBound.branch row11897_block_011_covers row11897_large_right_right_right_covers

def row11897_large_right : List Nat := row11897_large_right_left ++ row11897_large_right_right
theorem row11897_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11897 row11897_large_right :=
  Budget.RowUpperBound.branch row11897_large_right_left_covers row11897_large_right_right_covers

def row11897_large : List Nat := row11897_large_left ++ row11897_large_right
theorem row11897_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11897 row11897_large :=
  Budget.RowUpperBound.branch row11897_large_left_covers row11897_large_right_covers

def row11897 : List Nat := row11897_large
theorem row11897_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11897 row11897 := row11897_large_covers

def row11903_small : List Nat := []
theorem row11903_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11903 row11903_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11903_block_005 : List Nat := []
theorem row11903_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11903 row11903_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11903_block_006 : List Nat := []
def plan11903_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 31), .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .second, (.secondFactor 19), (.secondFactor 7), (.firstFactor 13), .first, .second, .first, (.secondFactor 17), .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 31), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, (.secondFactor 17), .first, (.secondFactor 13)]
set_option maxRecDepth 2048 in
theorem row11903_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11903 row11903_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11903_block_006
  · decide
  · rfl

def row11903_block_007 : List Nat := [12527]
def plan11903_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 23), (.firstFactor 11), (.secondFactor 17), .first, .first, .possible, (.firstFactor 13), .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 17), (.firstFactor 29), .first, .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 11), .first, .first, .first, (.firstFactor 31), .second, .first, (.firstFactor 13), .first, .first, (.secondFactor 13), (.firstFactor 23), .second, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 23), (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, (.firstFactor 29), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 23), .first, (.secondFactor 37), (.secondFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 31), .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 19), (.secondFactor 13), (.secondFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 37), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11903_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11903 row11903_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11903_block_007
  · decide
  · rfl

def row11903_block_008 : List Nat := []
theorem row11903_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11903 row11903_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11903_block_009 : List Nat := []
theorem row11903_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11903 row11903_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11903_block_010 : List Nat := []
theorem row11903_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11903 row11903_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11903_block_011 : List Nat := []
theorem row11903_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11903 row11903_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11903_block_012 : List Nat := []
theorem row11903_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11903 row11903_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11903_block_013 : List Nat := []
theorem row11903_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11903 row11903_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11903_large_left_left : List Nat := row11903_small ++ row11903_block_005
theorem row11903_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11903 row11903_large_left_left :=
  Budget.RowUpperBound.branch row11903_small_covers row11903_block_005_covers

def row11903_large_left_right_right : List Nat := row11903_block_007 ++ row11903_block_008
theorem row11903_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11903 row11903_large_left_right_right :=
  Budget.RowUpperBound.branch row11903_block_007_covers row11903_block_008_covers

def row11903_large_left_right : List Nat := row11903_block_006 ++ row11903_large_left_right_right
theorem row11903_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11903 row11903_large_left_right :=
  Budget.RowUpperBound.branch row11903_block_006_covers row11903_large_left_right_right_covers

def row11903_large_left : List Nat := row11903_large_left_left ++ row11903_large_left_right
theorem row11903_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11903 row11903_large_left :=
  Budget.RowUpperBound.branch row11903_large_left_left_covers row11903_large_left_right_covers

def row11903_large_right_left : List Nat := row11903_block_009 ++ row11903_block_010
theorem row11903_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11903 row11903_large_right_left :=
  Budget.RowUpperBound.branch row11903_block_009_covers row11903_block_010_covers

def row11903_large_right_right_right : List Nat := row11903_block_012 ++ row11903_block_013
theorem row11903_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11903 row11903_large_right_right_right :=
  Budget.RowUpperBound.branch row11903_block_012_covers row11903_block_013_covers

def row11903_large_right_right : List Nat := row11903_block_011 ++ row11903_large_right_right_right
theorem row11903_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11903 row11903_large_right_right :=
  Budget.RowUpperBound.branch row11903_block_011_covers row11903_large_right_right_right_covers

def row11903_large_right : List Nat := row11903_large_right_left ++ row11903_large_right_right
theorem row11903_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11903 row11903_large_right :=
  Budget.RowUpperBound.branch row11903_large_right_left_covers row11903_large_right_right_covers

def row11903_large : List Nat := row11903_large_left ++ row11903_large_right
theorem row11903_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11903 row11903_large :=
  Budget.RowUpperBound.branch row11903_large_left_covers row11903_large_right_covers

def row11903 : List Nat := row11903_large
theorem row11903_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11903 row11903 := row11903_large_covers

end ProjectEulerSolutions.PrimePairLarge
