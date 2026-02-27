CREATE TABLE fact_ctr_event (
    event_sk BIGINT PRIMARY KEY,
    inventory_id BIGINT,
    day_of_week INT,
    hour INT,
    gender INT,
    age_group INT,
    seq BIGINT,
    clicked INT
);

CREATE TABLE agg_inventory_ctr_time (
    inventory_id BIGINT,
    day_of_week INT,
    hour INT,
    impression_cnt BIGINT,
    click_cnt BIGINT,
    ctr FLOAT,
    PRIMARY KEY (inventory_id, day_of_week, hour)
);
