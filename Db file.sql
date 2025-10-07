DROP DATABASE IF EXISTS healthcare;
CREATE DATABASE healthcare;

USE healthcare;

create table if not exists user_details (
id INT AUTO_INCREMENT PRIMARY KEY,
user_name varchar(50) not null, 
email_id varchar(100) not null unique,
password varchar(100) not null
);
 
select * from user_details;


